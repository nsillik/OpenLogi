//! OS input-event synthesis for each [`Action`], split out of openlogi-core so
//! the core schema stays platform- and IO-free.
//!
//! [`execute`] is the single entry point: it dispatches to the per-platform
//! synthesiser (`macos::execute` / `linux::execute` / `windows::execute`), each
//! of which translates an [`Action`] into the native event(s) — CGEvent/NSEvent
//! on macOS, uinput/D-Bus on Linux, SendInput on Windows.

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
use std::collections::HashMap;
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
use std::sync::{LazyLock, Mutex, PoisonError};

use openlogi_core::binding::{Action, HoldKind, KeyCombo, KeyboardUsage};
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
use openlogi_core::binding::{Script, WorkflowStep};
use openlogi_core::scroll::ScrollDelta;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
use linux as platform;
#[cfg(target_os = "macos")]
use macos as platform;
#[cfg(target_os = "windows")]
use windows as platform;

/// Which isolated edge of a held keyboard chord to synthesize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyPhase {
    Down,
    Up,
}

/// One physical keyboard output shared by held outputs.
///
/// Logical Cmd and Ctrl are distinct on macOS. Cmd aliases Ctrl on Linux and
/// Windows, so ownership is counted after that platform mapping is resolved;
/// Command itself only ever has edges posted on macOS.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum HeldKey {
    // macOS-only concept: a chord carrying Command aliases to Control upstream
    // on Linux and Windows (see `chord_keys`), so nothing constructs it there.
    #[cfg(target_os = "macos")]
    Command,
    Control,
    Shift,
    Alt,
    Key(KeyboardUsage),
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
#[derive(Debug, Default, PartialEq, Eq)]
struct HoldTransition {
    up: Vec<HeldKey>,
    down: Vec<HeldKey>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct HeldModifiers(u8);

#[cfg(target_os = "macos")]
impl HeldModifiers {
    fn set(&mut self, key: HeldKey, held: bool) {
        let Some(mask) = Self::mask(key) else {
            return;
        };
        if held {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }

    fn contains(self, key: HeldKey) -> bool {
        Self::mask(key).is_some_and(|mask| self.0 & mask != 0)
    }

    fn mask(key: HeldKey) -> Option<u8> {
        match key {
            HeldKey::Command => Some(1 << 0),
            HeldKey::Control => Some(1 << 1),
            HeldKey::Shift => Some(1 << 2),
            HeldKey::Alt => Some(1 << 3),
            HeldKey::Key(_) => None,
        }
    }
}

/// Reference counts for the physical keys held by active outputs.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
#[derive(Default)]
struct HeldKeyOwners {
    owners: HashMap<HeldKey, usize>,
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
impl HeldKeyOwners {
    /// Release `released` and press `pressed` as one batch, reporting the edges
    /// whose ownership crossed zero.
    ///
    /// The batch is what keeps a key steady: one still owned when the batch
    /// ends reports neither an up nor a down edge, so a hold that takes over a
    /// modifier another hold already has down does not flicker it. Order is
    /// meaningful between batches, not within one.
    fn transition(&mut self, released: &[HeldKey], pressed: &[HeldKey]) -> HoldTransition {
        let before = self.owners.clone();

        for key in released {
            match self.owners.get_mut(key) {
                Some(owners) if *owners > 1 => *owners -= 1,
                Some(_) => {
                    self.owners.remove(key);
                }
                None => {}
            }
        }
        for key in pressed {
            *self.owners.entry(*key).or_default() += 1;
        }

        HoldTransition {
            up: released
                .iter()
                .copied()
                .filter(|key| before.contains_key(key) && !self.owners.contains_key(key))
                .collect(),
            down: pressed
                .iter()
                .copied()
                .filter(|key| !before.contains_key(key) && self.owners.contains_key(key))
                .collect(),
        }
    }

    #[cfg(target_os = "macos")]
    fn modifiers(&self) -> HeldModifiers {
        let mut modifiers = HeldModifiers::default();
        for key in [
            HeldKey::Command,
            HeldKey::Control,
            HeldKey::Shift,
            HeldKey::Alt,
        ] {
            modifiers.set(key, self.owners.contains_key(&key));
        }
        modifiers
    }
}

#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
static HELD_KEY_OWNERS: LazyLock<Mutex<HeldKeyOwners>> =
    LazyLock::new(|| Mutex::new(HeldKeyOwners::default()));

/// The physical keys the platform's application switcher owns.
///
/// Command on macOS — the ⌘Tab switcher commits when ⌘ comes up — and Alt on
/// Linux and Windows, whose native switcher is Alt+Tab; a ⇥ tap under this
/// modifier is what opens it. macOS reads the modifier from here rather than
/// restating which key this platform's switcher uses, and the slice is empty on
/// a target with no switcher to open.
const SWITCHER_KEYS: &[HeldKey] = cfg_select! {
    target_os = "macos" => {
        &[HeldKey::Command]
    }
    target_os = "linux" => {
        &[HeldKey::Alt]
    }
    target_os = "windows" => {
        &[HeldKey::Alt]
    }
    _ => {
        &[]
    }
};

/// The physical keys one held output owns.
fn held_keys(kind: &HoldKind) -> Vec<HeldKey> {
    match kind {
        HoldKind::Chord(combo) => chord_keys(combo),
        HoldKind::AppSwitcher => SWITCHER_KEYS.to_vec(),
    }
}

/// The physical keys a chord owns.
///
/// Logical Cmd and Ctrl are distinct on macOS. Cmd aliases Ctrl on Linux and
/// Windows, so the mapping resolves before ownership is counted.
fn chord_keys(combo: &KeyCombo) -> Vec<HeldKey> {
    let mut keys = Vec::with_capacity(4);
    #[cfg(target_os = "macos")]
    if combo.has_command() {
        keys.push(HeldKey::Command);
    }
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if combo.has_command() || combo.has_control() {
        keys.push(HeldKey::Control);
    }
    #[cfg(target_os = "macos")]
    if combo.has_control() {
        keys.push(HeldKey::Control);
    }
    if combo.has_shift() {
        keys.push(HeldKey::Shift);
    }
    if combo.has_option() {
        keys.push(HeldKey::Alt);
    }
    keys.push(HeldKey::Key(combo.key()));
    keys
}

/// A shortcut-table entry, parsed once into the chord it names. The tables are
/// hand-written constants, so a parse failure is a programming error.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn parse_shortcut(text: &str) -> KeyCombo {
    text.parse()
        .unwrap_or_else(|error| unreachable!("hardcoded shortcut table entry {text:?}: {error}"))
}

/// Run a script off the caller's thread: a shell command, an AppleScript or a
/// workflow can take seconds, and the caller is the input hook.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn dispatch_script(script: Script<'_>) {
    match script {
        Script::AppleScript(src) => {
            let src = src.to_string();
            std::thread::spawn(move || platform::run_apple_script(&src));
        }
        Script::ShellCommand(cmd) => {
            let cmd = cmd.to_string();
            std::thread::spawn(move || platform::run_shell_command(&cmd));
        }
        Script::Workflow(steps) => {
            let steps = steps.to_vec();
            std::thread::spawn(move || run_workflow(&steps));
        }
    }
}

/// Run workflow steps in order on the current (worker) thread, so a `Delay`
/// never stalls the event tap. Each step is one call into the platform
/// backend; a backend that cannot perform a step logs and moves on.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
fn run_workflow(steps: &[WorkflowStep]) {
    for step in steps {
        match step {
            WorkflowStep::TypeText(text) => platform::type_text(text),
            WorkflowStep::PressKey(combo) => platform::press_combo(combo),
            WorkflowStep::Delay { millis } => {
                std::thread::sleep(std::time::Duration::from_millis(*millis));
            }
            WorkflowStep::RunAppleScript(src) => platform::run_apple_script(src),
            WorkflowStep::RunShellCommand(cmd) => platform::run_shell_command(cmd),
        }
    }
}

/// Synthesise the OS-level event for `action`.
///
/// On macOS, key events are posted via `CGEventPost(kCGHIDEventTap, …)`
/// using virtual key codes from the standard US keyboard layout, and the
/// `LeftClick`/`RightClick`/`MiddleClick` variants synthesise a mouse click
/// at the current cursor location. The WindowServer actions (`MissionControl`,
/// `AppExpose`, `ShowDesktop`, `LaunchpadShow`) are posted straight to the
/// Dock via `CoreDockSendNotification`. Device-side actions (`CycleDpiPresets`,
/// `SetDpiPreset`, `ToggleSmartShift`) have no CGEvent equivalent and are
/// handled at the hook/HID layer, logging a trace here.
///
/// On Linux, key and scroll events are injected via a lazily-created `uinput`
/// virtual device. Mouse clicks inject `BTN_*` events. macOS-only window
/// manager actions (`MissionControl`, `AppExpose`, `ShowDesktop`,
/// `LaunchpadShow`) have no universal Linux equivalent and are silently
/// skipped (debug-logged). `CustomShortcut` maps macOS `kVK_*` codes to
/// Linux key codes; macOS Cmd maps to Ctrl.
///
/// On Windows, key and mouse events are synthesised via `SendInput`. The
/// macOS window-manager actions map to their Windows equivalents (e.g.
/// `MissionControl` → Win+Tab, `ShowDesktop` → Win+D); `CustomShortcut`
/// maps macOS `kVK_*` codes to Windows virtual-key codes, with Cmd mapped to
/// Ctrl.
///
/// On other platforms a warning is logged and the function returns
/// immediately — the binary compiles clean on all targets.
///
/// # Manual verification
///
/// `execute` is intentionally excluded from the automated test suite because
/// it would need to intercept the OS event queue. Smoke-test it manually:
/// bind a button to any action in the GUI and confirm the expected system event
/// fires when the button is pressed (or use the `inject_action` example).
pub fn execute(action: &Action) {
    if let Action::OpenApplication(target) = action {
        let expanded = shellexpand::tilde(target.path());
        if let Err(error) = opener::open(expanded.as_ref()) {
            tracing::warn!(
                %error,
                path = target.path(),
                "could not open configured application, folder, or URL"
            );
        }
        return;
    }

    cfg_select! {
        target_os = "macos" => {
            macos::execute(action);
        }
        target_os = "linux" => {
            linux::execute(action);
        }
        target_os = "windows" => {
            windows::execute(action);
        }
        _ => {
            tracing::warn!(
                action = action.label(),
                "execute unsupported on this platform"
            );
        }
    }
}

/// One synthetic held keyboard output, released exactly once when dropped.
///
/// Keep this value with the physical press lifecycle. Repointing it at another
/// output ([`HeldOutput::retarget`]) preserves the physical keys the two
/// outputs share; cancellation, shutdown, and unwinding all release the current
/// output through [`Drop`].
#[must_use = "dropping the held output immediately releases its synthetic keys"]
pub struct HeldOutput {
    kind: HoldKind,
}

impl HeldOutput {
    /// Repoint this held output at `kind`, for a repeat trigger, a long-press
    /// threshold, or a per-app rebind re-firing a held action on the same
    /// physical press.
    pub fn retarget(&mut self, kind: HoldKind) {
        let previous = std::mem::replace(&mut self.kind, kind);
        for step in retarget_steps(&previous, &self.kind) {
            step.post();
        }
    }
}

/// One posted change to the synthetic keyboard output.
#[derive(Debug, PartialEq, Eq)]
enum HoldStep {
    /// One ownership batch: release these keys, then press those.
    Transition {
        released: Vec<HeldKey>,
        pressed: Vec<HeldKey>,
    },
    /// Open the application switcher with one ⇥ tap, under the modifier the
    /// transition before it left down.
    TapAppSwitcher,
}

impl HoldStep {
    fn post(self) {
        match self {
            Self::Transition { released, pressed } => hold_transition(&released, &pressed),
            Self::TapAppSwitcher => tap_app_switcher(),
        }
    }
}

/// The changes that repoint a live press from `previous` to `next`, in the
/// order they post.
///
/// [`HeldKeyOwners::transition`] releases before it presses and reports no edge
/// for a key that survives the batch, so which changes share a batch is the
/// whole contract here:
///
/// - chord → chord, and chord → switcher: one batch, which retires what only
///   the old output held and presses what only the new one holds. A key both
///   hold — the switcher's modifier included — keeps its down edge.
/// - switcher → switcher: nothing at all. The switcher is open and its ⇥ tap
///   has posted; a second tap would step the selection instead of holding it.
/// - switcher → chord: release, then press. The switcher commits on its
///   modifier's up edge, so that edge has to land before a chord sharing the
///   modifier takes it over — one batch would leave the switcher open.
///
/// The ⇥ tap is last, because it is what opens the switcher: a key of the
/// outgoing chord still down under it (⇧, say) would change the tap's chord and
/// step the selection backwards.
fn retarget_steps(previous: &HoldKind, next: &HoldKind) -> Vec<HoldStep> {
    match (previous, next) {
        (HoldKind::AppSwitcher, HoldKind::AppSwitcher) => Vec::new(),
        (HoldKind::AppSwitcher, HoldKind::Chord(_)) => vec![
            HoldStep::Transition {
                released: held_keys(previous),
                pressed: Vec::new(),
            },
            HoldStep::Transition {
                released: Vec::new(),
                pressed: held_keys(next),
            },
        ],
        (HoldKind::Chord(_), HoldKind::Chord(_)) => vec![HoldStep::Transition {
            released: held_keys(previous),
            pressed: held_keys(next),
        }],
        (HoldKind::Chord(_), HoldKind::AppSwitcher) => vec![
            HoldStep::Transition {
                released: held_keys(previous),
                pressed: held_keys(next),
            },
            HoldStep::TapAppSwitcher,
        ],
    }
}

impl Drop for HeldOutput {
    fn drop(&mut self) {
        hold_transition(&held_keys(&self.kind), &[]);
    }
}

/// Press the output `kind` names and return its release lease.
///
/// Keep the returned [`HeldOutput`] until the physical press ends. Prefer
/// [`execute`] when the caller does not own a matching terminal event.
pub fn press_hold(kind: HoldKind) -> HeldOutput {
    // Construct the lease before posting the edge so unwinding from the
    // platform backend still balances any ownership transition it completed.
    let held = HeldOutput { kind };
    hold_transition(&[], &held_keys(&held.kind));
    if matches!(held.kind, HoldKind::AppSwitcher) {
        // The switcher's modifier is down; one ⇥ tap opens the switcher, which
        // stays open until this lease is dropped.
        tap_app_switcher();
    }
    held
}

/// Open the application switcher with one ⇥ tap, under the modifier its lease
/// already holds down.
///
/// The modifier itself is not pressed here: [`held_keys`] posted its edge, and
/// each backend posts only the edges that changed.
fn tap_app_switcher() {
    cfg_select! {
        target_os = "macos" => {
            macos::tap_app_switcher();
        }
        target_os = "linux" => {
            linux::tap_app_switcher();
        }
        target_os = "windows" => {
            windows::tap_app_switcher();
        }
        _ => {}
    }
}

fn hold_transition(released: &[HeldKey], pressed: &[HeldKey]) {
    cfg_select! {
        target_os = "macos" => {
            let mut owners = HELD_KEY_OWNERS.lock().unwrap_or_else(PoisonError::into_inner);
            // `HeldKeyOwners::owners` is the only persistent modifier state.
            // This bitmask is an event-ordering cursor: derive it from the map
            // while holding the same mutex, advance it through the exact
            // transition edges, then prove it reached the map's post-transition
            // state.
            let modifiers = owners.modifiers();
            let transition = owners.transition(released, pressed);
            let modifiers = macos::hold_keys(&transition.up, KeyPhase::Up, modifiers);
            let modifiers = macos::hold_keys(&transition.down, KeyPhase::Down, modifiers);
            debug_assert_eq!(modifiers, owners.modifiers());
        }
        target_os = "linux" => {
            let mut owners = HELD_KEY_OWNERS.lock().unwrap_or_else(PoisonError::into_inner);
            let transition = owners.transition(released, pressed);
            linux::hold_keys(&transition.up, KeyPhase::Up);
            linux::hold_keys(&transition.down, KeyPhase::Down);
        }
        target_os = "windows" => {
            let mut owners = HELD_KEY_OWNERS.lock().unwrap_or_else(PoisonError::into_inner);
            let transition = owners.transition(released, pressed);
            windows::hold_keys(&transition.up, KeyPhase::Up);
            windows::hold_keys(&transition.down, KeyPhase::Down);
        }
        _ => {
            tracing::warn!(
                "held shortcut output unsupported on this platform"
            );
        }
    }
}

/// Navigate Safari backwards or forwards using `AXPress` on its toolbar
/// button's stable Accessibility identifier.
///
/// Pass the Safari process captured when the button press arrived. The call
/// returns `false` if that process is no longer frontmost or the frontmost app
/// is not Safari.
/// No-op (returns `false`) on non-macOS platforms.
#[must_use]
pub fn ax_navigate_browser(pid: i32, forward: bool) -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::ax_browser_navigate(forward, pid)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (pid, forward);
        false
    }
}

/// Integer scroll units ready for a platform API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct QuantizedScroll {
    x: i32,
    y: i32,
}

/// Carries fractional platform units across frames so rounding never changes
/// the cumulative distance.
#[derive(Default)]
struct ScrollQuantizer {
    residual_x: f64,
    residual_y: f64,
}

impl ScrollQuantizer {
    fn quantize(&mut self, delta: ScrollDelta, units_per_input: f64) -> QuantizedScroll {
        QuantizedScroll {
            x: quantize_axis(&mut self.residual_x, delta.x(), units_per_input),
            y: quantize_axis(&mut self.residual_y, delta.y(), units_per_input),
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded value is clamped to the i32 range before conversion"
)]
fn quantize_axis(residual: &mut f64, input: f64, units_per_input: f64) -> i32 {
    let exact = input.mul_add(units_per_input, *residual);
    let rounded = exact
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX));
    let output = rounded as i32;
    *residual = exact - f64::from(output);
    output
}

/// Synthesise a typed scroll distance at the current focus.
///
/// Fractional wheel ticks are retained until the platform can represent them,
/// so a sequence of high-resolution frames preserves its cumulative distance.
/// Non-finite input is rejected at this I/O boundary.
pub fn post_scroll(delta: ScrollDelta) {
    if !delta.is_finite() || (delta.x() == 0.0 && delta.y() == 0.0) {
        return;
    }
    cfg_select! {
        target_os = "macos" => {
            macos::post_scroll(delta);
        }
        target_os = "linux" => {
            linux::post_scroll(delta);
        }
        target_os = "windows" => {
            windows::post_scroll(delta);
        }
        _ => {
            let _ = delta;
        }
    }
}

/// Lifecycle phase of one synthetic smooth-scroll frame.
///
/// macOS forwards this state to the scroll-wheel event so applications see a
/// balanced continuous gesture. Linux and Windows have no equivalent field;
/// there the phase is retained by the runtime contract but only the frame's
/// distance is injected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothScrollPhase {
    /// First output frame of a new animation.
    Began,
    /// An intermediate output frame, including frames after retargeting.
    Changed,
    /// Final frame, carrying any correction needed to reach the exact target.
    Ended,
    /// The capture source ended before the animation reached its target.
    Cancelled,
}

/// Synthesise one frame of a finite smooth-scroll animation.
///
/// On macOS wheel ticks become continuous pixel events at ten points per tick,
/// matching the line/point relationship carried in native continuous events.
/// Other platforms preserve fractional wheel ticks through their native
/// high-resolution output. Non-finite distance is rejected at this I/O
/// boundary; zero-distance terminal frames remain meaningful on macOS.
pub fn post_smooth_scroll(delta: ScrollDelta, phase: SmoothScrollPhase) {
    if !delta.is_finite() {
        return;
    }
    cfg_select! {
        target_os = "macos" => {
            macos::post_smooth_scroll(delta, phase);
        }
        _ => {
            let _ = phase;
            post_scroll(delta);
        }
    }
}

/// Return the `/dev/input/eventN` node for the action-injector uinput device,
/// initialising it if needed.
///
/// Intended for debugging and manual smoke-testing (e.g. attaching `evtest`
/// before firing [`execute`]). Returns `None` on non-Linux platforms or
/// when the device could not be created (e.g. `/dev/uinput` not writable).
#[cfg(target_os = "linux")]
#[must_use]
pub fn action_device_path() -> Option<std::path::PathBuf> {
    linux::device_node()
}

/// Stamped into the `EVENT_SOURCE_USER_DATA` field of every mouse event
/// [`execute`] synthesizes on macOS, so OpenLogi's own `CGEventTap` can
/// recognize and skip its own injections. Without it, a gesture/button action
/// that posts a mouse button (e.g. a remapped `MiddleClick`) would re-enter the
/// hook — and for a gesture button, be misread as a fresh hold, looping. The
/// value is arbitrary but distinctive ("OLGI"); real events carry `0` here.
pub const SYNTHETIC_EVENT_USER_DATA: i64 = 0x4F4C_4749;

/// Translate a platform-neutral USB HID keyboard usage to a Win32 virtual key.
// Not `expect`: the lint fires in the `--lib` build and not in the `--test`
// one, so an expectation is always unfulfilled for one of them.
#[cfg_attr(
    not(target_os = "windows"),
    expect(clippy::allow_attributes, reason = "see above"),
    allow(dead_code, reason = "called only by the Windows backend")
)]
fn hid_usage_to_windows(usage: u8) -> Option<u16> {
    match usage {
        0x04..=0x1d => Some(u16::from(b'A' + usage - 0x04)),
        0x1e..=0x26 => Some(u16::from(b'1' + usage - 0x1e)),
        0x27 => Some(u16::from(b'0')),
        0x3a..=0x45 => Some(0x70 + u16::from(usage - 0x3a)),
        0x68..=0x6f => Some(0x7c + u16::from(usage - 0x68)),
        0x28 => Some(0x0d),
        0x29 => Some(0x1b),
        0x2a => Some(0x08),
        0x2b => Some(0x09),
        0x2c => Some(0x20),
        0x2d => Some(0xbd),
        0x2e => Some(0xbb),
        0x2f => Some(0xdb),
        0x30 => Some(0xdd),
        0x31 => Some(0xdc),
        0x33 => Some(0xba),
        0x34 => Some(0xde),
        0x35 => Some(0xc0),
        0x36 => Some(0xbc),
        0x37 => Some(0xbe),
        0x38 => Some(0xbf),
        0x4a => Some(0x24),
        0x4b => Some(0x21),
        0x4c => Some(0x2e),
        0x4d => Some(0x23),
        0x4e => Some(0x22),
        0x4f => Some(0x27),
        0x50 => Some(0x25),
        0x51 => Some(0x28),
        0x52 => Some(0x26),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use openlogi_core::scroll::ScrollDelta;

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    use openlogi_core::binding::{HoldKind, KeyCombo};

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    use super::{HeldKey, HeldKeyOwners, HoldStep, HoldTransition, chord_keys, held_keys};
    use super::{QuantizedScroll, ScrollQuantizer};

    /// Synthetic high-resolution input: eight eighth-ticks must total exactly
    /// one Windows/Linux wheel detent (120 raw units). This is deterministic
    /// model data, not a hardware capture.
    #[test]
    fn fractional_frames_preserve_cumulative_wheel_distance() {
        let mut quantizer = ScrollQuantizer::default();
        let total = (0..8)
            .map(|_| {
                quantizer
                    .quantize(ScrollDelta::wheel_ticks(0.0, 0.125), 120.0)
                    .y
            })
            .sum::<i32>();
        assert_eq!(total, 120);
    }

    #[test]
    fn opposing_fractional_input_cancels_without_rounding_drift() {
        let mut quantizer = ScrollQuantizer::default();
        let forward = quantizer.quantize(ScrollDelta::wheel_ticks(0.25, 0.0), 120.0);
        let backward = quantizer.quantize(ScrollDelta::wheel_ticks(-0.25, 0.0), 120.0);
        assert_eq!(forward, QuantizedScroll { x: 30, y: 0 });
        assert_eq!(backward, QuantizedScroll { x: -30, y: 0 });
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    fn combo(label: &str) -> KeyCombo {
        label.parse().expect("test shortcut must be valid")
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn shared_control_stays_down_until_its_last_chord_ends() {
        let control_a = combo("Ctrl+A");
        let control_b = combo("Ctrl+B");
        let mut owners = HeldKeyOwners::default();

        assert_eq!(
            owners.transition(&[], &chord_keys(&control_a)),
            HoldTransition {
                up: vec![],
                down: vec![HeldKey::Control, HeldKey::Key(control_a.key())],
            }
        );
        assert_eq!(
            owners.transition(&[], &chord_keys(&control_b)),
            HoldTransition {
                up: vec![],
                down: vec![HeldKey::Key(control_b.key())],
            }
        );
        assert_eq!(
            owners.transition(&chord_keys(&control_a), &[]),
            HoldTransition {
                up: vec![HeldKey::Key(control_a.key())],
                down: vec![],
            }
        );
        assert_eq!(
            owners.transition(&chord_keys(&control_b), &[]),
            HoldTransition {
                up: vec![HeldKey::Control, HeldKey::Key(control_b.key())],
                down: vec![],
            }
        );
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn command_and_control_share_one_physical_output() {
        let command_a = combo("Cmd+A");
        let control_b = combo("Ctrl+B");
        let mut owners = HeldKeyOwners::default();

        owners.transition(&[], &chord_keys(&command_a));
        assert_eq!(
            owners.transition(&[], &chord_keys(&control_b)),
            HoldTransition {
                up: vec![],
                down: vec![HeldKey::Key(control_b.key())],
            }
        );
        assert_eq!(
            owners.transition(&chord_keys(&command_a), &[]),
            HoldTransition {
                up: vec![HeldKey::Key(command_a.key())],
                down: vec![],
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn command_and_control_are_distinct_physical_outputs() {
        let command_a = combo("Cmd+A");
        let control_b = combo("Ctrl+B");
        let mut owners = HeldKeyOwners::default();

        owners.transition(&[], &chord_keys(&command_a));
        assert_eq!(
            owners.transition(&[], &chord_keys(&control_b)),
            HoldTransition {
                up: vec![],
                down: vec![HeldKey::Control, HeldKey::Key(control_b.key())],
            }
        );
        assert_eq!(
            owners.transition(&chord_keys(&command_a), &[]),
            HoldTransition {
                up: vec![HeldKey::Command, HeldKey::Key(command_a.key())],
                down: vec![],
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn shared_command_stays_down_until_its_last_chord_ends() {
        let command_a = combo("Cmd+A");
        let command_b = combo("Cmd+B");
        let mut owners = HeldKeyOwners::default();

        owners.transition(&[], &chord_keys(&command_a));
        assert_eq!(
            owners.transition(&[], &chord_keys(&command_b)),
            HoldTransition {
                up: vec![],
                down: vec![HeldKey::Key(command_b.key())],
            }
        );
        assert_eq!(
            owners.transition(&chord_keys(&command_a), &[]),
            HoldTransition {
                up: vec![HeldKey::Key(command_a.key())],
                down: vec![],
            }
        );
        assert_eq!(
            owners.transition(&chord_keys(&command_b), &[]),
            HoldTransition {
                up: vec![HeldKey::Command, HeldKey::Key(command_b.key())],
                down: vec![],
            }
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn replacement_preserves_shared_physical_outputs() {
        let old = combo("Ctrl+A");
        let new = combo("Ctrl+B");
        let mut owners = HeldKeyOwners::default();

        owners.transition(&[], &chord_keys(&old));
        assert_eq!(
            owners.transition(&chord_keys(&old), &chord_keys(&new)),
            HoldTransition {
                up: vec![HeldKey::Key(old.key())],
                down: vec![HeldKey::Key(new.key())],
            }
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn the_switcher_hold_owns_the_platform_modifier_alone() {
        #[cfg(target_os = "macos")]
        let expected = vec![HeldKey::Command];
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let expected = vec![HeldKey::Alt];

        assert_eq!(held_keys(&HoldKind::AppSwitcher), expected);
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn retargeting_an_open_switcher_at_a_chord_commits_it_first() {
        let switcher = HoldKind::AppSwitcher;
        // Carries both primary modifiers, so the chord holds whatever key this
        // platform's switcher holds: Cmd aliases Control on Linux and Windows,
        // where the switcher holds Alt instead.
        let chord = HoldKind::Chord(combo("Cmd+Alt+Shift+L"));
        assert!(
            held_keys(&chord)
                .iter()
                .any(|key| held_keys(&switcher).contains(key)),
            "the two outputs have to share a key for this to be the case"
        );

        // Two batches, in this order: the first one's up edge is the
        // switcher's commit, and the chord presses after it. One combined
        // batch would report no edge for the shared modifier and leave the
        // switcher open under the chord.
        assert_eq!(
            super::retarget_steps(&switcher, &chord),
            vec![
                HoldStep::Transition {
                    released: held_keys(&switcher),
                    pressed: Vec::new(),
                },
                HoldStep::Transition {
                    released: Vec::new(),
                    pressed: held_keys(&chord),
                },
            ]
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
    #[test]
    fn opening_the_switcher_taps_after_the_batch_that_frees_its_modifier() {
        let switcher = HoldKind::AppSwitcher;
        let chord = HoldKind::Chord(combo("Cmd+Alt+Shift+L"));

        // One batch — the modifier the chord already holds keeps its down edge
        // — and the ⇥ tap after it, never over the keys only the chord held.
        assert_eq!(
            super::retarget_steps(&chord, &switcher),
            vec![
                HoldStep::Transition {
                    released: held_keys(&chord),
                    pressed: held_keys(&switcher),
                },
                HoldStep::TapAppSwitcher,
            ]
        );
        // Already open: nothing to post, and no second tap to step it.
        assert_eq!(super::retarget_steps(&switcher, &switcher), Vec::new());
    }

    #[test]
    fn hid_usages_map_across_windows_key_categories() {
        use super::hid_usage_to_windows;

        assert_eq!(hid_usage_to_windows(0x04), Some(0x41)); // A
        assert_eq!(hid_usage_to_windows(0x1e), Some(0x31)); // 1
        assert_eq!(hid_usage_to_windows(0x3a), Some(0x70)); // F1
        assert_eq!(hid_usage_to_windows(0x6f), Some(0x83)); // F20
        assert_eq!(hid_usage_to_windows(0x50), Some(0x25)); // Left
        assert_eq!(hid_usage_to_windows(0x2c), Some(0x20)); // Space
        assert_eq!(hid_usage_to_windows(0x33), Some(0xba)); // Semicolon
        assert_eq!(hid_usage_to_windows(0xff), None);
    }
}
