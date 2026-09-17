"""
Context Detector
Determines the most appropriate output context (EMAIL, CHAT, NOTE, …)
by combining:
 - Detected intent from the NLP layer
 - Active application name
 - Optional user-selected mode override
"""

import logging

logger = logging.getLogger(__name__)

# Maps active application names (lower-case substrings) → context
APP_CONTEXT_MAP: dict[str, str] = {
    "mail":     "EMAIL",
    "outlook":  "EMAIL",
    "gmail":    "EMAIL",
    "thunderbird": "EMAIL",
    "slack":    "CHAT",
    "teams":    "CHAT",
    "discord":  "CHAT",
    "whatsapp": "CHAT",
    "telegram": "CHAT",
    "notion":   "NOTE",
    "obsidian": "NOTE",
    "notes":    "NOTE",
    "code":     "CODE",       # VS Code
    "pycharm":  "CODE",
    "intellij": "CODE",
    "xcode":    "CODE",
    "chrome":   "SEARCH",
    "firefox":  "SEARCH",
    "safari":   "SEARCH",
    "edge":     "SEARCH",
}

# Maps intent labels → context (used when no app match)
INTENT_CONTEXT_MAP: dict[str, str] = {
    "EMAIL":    "EMAIL",
    "CHAT":     "CHAT",
    "NOTE":     "NOTE",
    "SEARCH":   "SEARCH",
    "CODE":     "CODE",
    "REMINDER": "NOTE",
    "QUESTION": "SEARCH",
}

DEFAULT_CONTEXT = "GENERAL"


class ContextDetector:
    """
    Determines the output context for a processed utterance.

    Priority order:
    1. Explicit user-selected mode override
    2. Active application mapping
    3. Intent classification mapping
    4. DEFAULT_CONTEXT ('GENERAL')
    """

    def detect(
        self,
        nlp_result: dict,
        active_app: str = "",
        user_mode: str = "",
    ) -> str:
        """
        Determine the context string.

        Parameters
        ----------
        nlp_result : dict
            Output from NLPPipeline.process().
        active_app : str
            Name / title of the currently focused application.
        user_mode : str
            Optional override selected in the UI.

        Returns
        -------
        str  — One of: EMAIL, CHAT, NOTE, SEARCH, CODE, GENERAL
        """
        # 1. Explicit override
        if user_mode and user_mode.upper() != "AUTO":
            logger.debug("Context: user override → %s", user_mode.upper())
            return user_mode.upper()

        # 2. Active application
        app_lower = active_app.lower()
        for keyword, ctx in APP_CONTEXT_MAP.items():
            if keyword in app_lower:
                logger.debug("Context: app match '%s' → %s", keyword, ctx)
                return ctx

        # 3. Intent mapping
        intent_label: str = (
            nlp_result.get("intent", {}).get("label", "") if isinstance(nlp_result, dict) else ""
        )
        if intent_label in INTENT_CONTEXT_MAP:
            ctx = INTENT_CONTEXT_MAP[intent_label]
            logger.debug("Context: intent '%s' → %s", intent_label, ctx)
            return ctx

        logger.debug("Context: default → %s", DEFAULT_CONTEXT)
        return DEFAULT_CONTEXT
