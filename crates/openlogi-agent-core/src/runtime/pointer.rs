//! Admission of pointer-selected actions, on the action worker, never the tap.

use openlogi_core::binding::{Action, Effect, NativeAction};
use openlogi_hook::PointerTarget;

use super::ActionDispatchTarget;

impl ActionDispatchTarget {
    pub(super) fn resolve(self, action: &Action) -> Option<Self> {
        let Self::Pointer(target) = self else {
            return Some(self);
        };
        let current = openlogi_hook::pointer_context();
        if !pointer_action_allowed(action, target, current.target, || {
            openlogi_hook::pointer_target_is_focused(target)
        }) {
            return None;
        }
        // Safari's existing AX implementation uses AXFocusedWindow. It is
        // admitted only after the exact hovered window passed the focus check.
        Some(match target {
            PointerTarget::Window { process_id, .. }
                if openlogi_hook::frontmost_safari_pid() == Some(process_id) =>
            {
                Self::SafariProcess(process_id)
            }
            _ => Self::Keyboard,
        })
    }
}

fn pointer_action_allowed(
    action: &Action,
    captured: PointerTarget,
    current: PointerTarget,
    is_focused: impl FnOnce() -> bool,
) -> bool {
    if captured != current
        || matches!(
            captured,
            PointerTarget::Unavailable | PointerTarget::Unsupported
        )
    {
        return false;
    }
    // Only an action whose output lands in the hovered window needs it focused
    // or is invalidated when that window goes away — see
    // `addresses_pointer_window`.
    !addresses_pointer_window(action)
        || (matches!(captured, PointerTarget::Window { .. }) && is_focused())
}

/// Whether this action's output is addressed at the window under the pointer.
///
/// One decision with two consumers, and they have to agree:
/// [`pointer_action_allowed`] refuses such an action once the captured window is
/// no longer hovered or focused, and `ButtonState::cancel_pointer_except` ends
/// such a press when the hovered window changes. An action that answers `false`
/// has no window to lose, so a pointer move must not disturb it — for the app
/// switcher, whose own panel becomes the hovered window the moment it opens,
/// that move would otherwise be the switcher's commit edge.
///
/// An effect qualifies when it hands its output to a window — a shortcut, a
/// chord, a typed string, a script, or App Exposé — rather than to the system at
/// large, the way media keys, desktop switching, the Actions Ring and the app
/// switcher do.
pub(super) fn addresses_pointer_window(action: &Action) -> bool {
    match action.effect() {
        Effect::Shortcut(_)
        | Effect::Key(_)
        | Effect::HeldKey(_)
        | Effect::Text(_)
        | Effect::Script(_)
        | Effect::Native(NativeAction::AppExpose) => true,
        Effect::None
        | Effect::Click(_)
        | Effect::AppSwitcher
        | Effect::Scroll { .. }
        | Effect::Media(_)
        | Effect::Native(_)
        | Effect::AgentSide => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BROWSER: PointerTarget = PointerTarget::Window {
        process_id: 41,
        window_id: 7,
    };
    const OTHER_WINDOW: PointerTarget = PointerTarget::Window {
        process_id: 41,
        window_id: 9,
    };

    #[test]
    fn desktop_switch_does_not_require_browser_focus_or_send_browser_navigation() {
        assert!(pointer_action_allowed(
            &Action::NextDesktop,
            PointerTarget::Desktop,
            PointerTarget::Desktop,
            || false
        ));
        assert!(!pointer_action_allowed(
            &Action::BrowserBack,
            PointerTarget::Desktop,
            PointerTarget::Desktop,
            || true
        ));
    }

    #[test]
    fn keyboard_effects_require_the_exact_hovered_window_to_have_focus() {
        let shortcut = "Ctrl+Tab".parse().expect("valid shortcut");
        for action in [
            Action::BrowserForward,
            Action::CustomShortcut(shortcut),
            Action::TypeText("hello".into()),
            Action::AppExpose,
        ] {
            assert!(!pointer_action_allowed(&action, BROWSER, BROWSER, || false));
            assert!(pointer_action_allowed(&action, BROWSER, BROWSER, || true));
            assert!(!pointer_action_allowed(
                &action,
                BROWSER,
                OTHER_WINDOW,
                || true
            ));
        }
    }

    #[test]
    fn stale_or_unknown_context_never_becomes_a_desktop_action() {
        for (captured, current) in [
            (BROWSER, PointerTarget::Desktop),
            (PointerTarget::Desktop, BROWSER),
            (PointerTarget::Unavailable, PointerTarget::Unavailable),
            (PointerTarget::Unsupported, PointerTarget::Unsupported),
        ] {
            assert!(!pointer_action_allowed(
                &Action::NextDesktop,
                captured,
                current,
                || true
            ));
        }
        assert!(pointer_action_allowed(
            &Action::VolumeUp,
            BROWSER,
            BROWSER,
            || false
        ));
    }
}
