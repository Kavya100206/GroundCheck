"""
src/policy.py

Phase 4 Component — Curated Troubleshooting Policy Reference.
Provides explicit, grounded standard operating procedures (SOPs) for SpotifyCares
customer support, pulled directly from genuine actionable resolution threads in the
strict visible-resolution training corpus (8,310 rows).

Strict Provenance Discipline:
All external URLs have been stripped. Steps reflect strictly verified in-app menu
navigations, device sequences, and documented public support policies present in the
1,173 concrete actionable training threads.
"""

from typing import Dict, Any, List, Optional

# Curated policy procedures grouped and deduplicated by operational intent
# Derived from 1,173 concrete actionable non-DM threads in src/resolution_corpus.csv
TROUBLESHOOTING_POLICIES: Dict[str, List[Dict[str, Any]]] = {
    "playback_issue": [
        {
            "id": "PB_01_BASIC_REFRESH",
            "name": "Session Refresh & Device Restart",
            "triggers": ["song skipping", "playback stuttering", "pauses unexpectedly", "music stops", "won't play", "skip"],
            "steps": [
                "1. Log out of your Spotify account.",
                "2. Fully close the Spotify app and restart your device.",
                "3. Log back in and test playback."
            ],
            "source_tweet_sample": "@159340 Thanks. Does logging out > restarting your device > logging back in help? /AN"
        },
        {
            "id": "PB_02_NETWORK_STREAMING",
            "name": "Network & Audio Streaming Quality Toggle",
            "triggers": ["buffering", "lag", "intermittent pauses", "slow streaming", "wifi cellular"],
            "steps": [
                "1. Test switching between Wi-Fi and mobile data (or restart your home Wi-Fi router).",
                "2. In Spotify Settings > Audio Quality, lower the streaming quality to Normal to prevent buffer underruns.",
                "3. If on desktop, toggle 'Hardware Acceleration' in Spotify Settings > Show Advanced Settings."
            ],
            "source_tweet_sample": "@116387 Thanks. Is it happening over 3G/4G, WiFi, or both? Does logging out, fully closing the app, and restarting your device help? /KC"
        },
        {
            "id": "PB_03_CONTENT_LICENSING",
            "name": "Greyed-out Track & Regional Availability",
            "triggers": ["greyed out", "song unplayable", "this song is not available", "broken album", "song missing"],
            "steps": [
                "1. Confirm if the track appears greyed out; content availability is determined by rights-holders and varies by region.",
                "2. In Settings > Playback, ensure 'Show unplayable songs' is enabled to verify status.",
                "3. Log out and back in to refresh local music licensing catalog tokens."
            ],
            "source_tweet_sample": "@130286 We can confirm that the track is no longer available. Sometimes content gets temporarily removed due to licensing agreements."
        },
        {
            "id": "PB_04_CLEAN_REINSTALL_PLAYBACK",
            "name": "Corrupted Audio Cache Clean Reinstall",
            "triggers": ["songs crash repeatedly", "playback frozen permanently", "audio distorted"],
            "steps": [
                "1. Uninstall the Spotify app from your device.",
                "2. Delete local cached app files to ensure corrupted audio streams are cleared.",
                "3. Reinstall the latest version of Spotify from your device's app store."
            ],
            "source_tweet_sample": "@135367 Gotcha. Can you try reinstalling the app? Let us know how it plays out /LM"
        }
    ],

    "app_technical": [
        {
            "id": "TECH_01_CLEAN_REINSTALL",
            "name": "Standard Clean Reinstallation Protocol",
            "triggers": ["app crash on launch", "black screen", "blank screen", "won't open", "freezes", "crash on startup", "crash"],
            "steps": [
                "1. Uninstall the Spotify app from your device.",
                "2. Manually delete residual cache files (Windows: %appdata%\\Spotify; Mac: ~/Library/Application Support/Spotify; Mobile: clear app storage).",
                "3. Restart the device.",
                "4. Reinstall a fresh copy from your device's official app store."
            ],
            "source_tweet_sample": "@224968 Thanks for letting us know. Can you try reinstalling the app? Follow the clean reinstall steps. /CE"
        },
        {
            "id": "TECH_02_CLEAR_CACHE",
            "name": "Application Storage & Cache Wipe",
            "triggers": ["app slow", "lagging", "high storage", "local files glitched", "clear cache"],
            "steps": [
                "1. Open Spotify and navigate to Settings > Storage.",
                "2. Tap 'Clear cache' (this will not delete downloaded offline playlists, but refreshes temporary index files).",
                "3. Restart the app."
            ],
            "source_tweet_sample": "@131454 Hmm. Does restarting your device help at all? Can you try clearing your cache? Keep us posted /PL"
        },
        {
            "id": "TECH_03_WEB_PLAYER",
            "name": "Web Player Browser Isolation",
            "triggers": ["web player crash", "web player won't load", "chrome web player", "browser frozen", "firefox"],
            "steps": [
                "1. Test the web player in a Private / Incognito browser window to bypass extension conflicts.",
                "2. Clear your browser's cached images and cookies.",
                "3. Ensure your browser's Protected Content / Widevine DRM playback setting is enabled."
            ],
            "source_tweet_sample": "@117526 Got it! Can you try using a different browser or an incognito window instead? Let us know how it goes /JU"
        },
        {
            "id": "TECH_04_DLL_RUNTIME",
            "name": "Windows Runtime / Corrupted Installer Repair",
            "triggers": ["dll missing", "api-ms-win", "error code", "installer error", "vcruntime"],
            "steps": [
                "1. Download the full standalone offline installer directly rather than using app store mirrors.",
                "2. Update pending Microsoft Visual C++ redistributable packages on Windows.",
                "3. Run the installer as Administrator."
            ],
            "source_tweet_sample": "@238740 For missing runtime DLL errors, please run a clean reinstall using our direct installer."
        }
    ],

    "account_login": [
        {
            "id": "AUTH_01_PASSWORD_RESET",
            "name": "Standard Self-Service Password Reset",
            "triggers": ["forgot password", "can't log in", "invalid password", "wrong password", "reset link"],
            "steps": [
                "1. Access the official password reset tool on the Spotify website using your registered email or username.",
                "2. Check your email inbox (including Spam/Junk folders) and follow the password reset link.",
                "3. If using Facebook login, verify your Facebook credentials or set an explicit device password."
            ],
            "source_tweet_sample": "@137386 You can request a password reset directly using your registered account email."
        },
        {
            "id": "AUTH_02_14_DAY_TRAVEL",
            "name": "14-Day Free Tier Overseas Travel Policy",
            "triggers": ["14 days abroad", "14 days overseas", "blocked abroad", "country mismatch login", "travelling"],
            "steps": [
                "1. Free Spotify accounts can only be used outside your registered home country for 14 days.",
                "2. To continue using Free abroad: log into your account overview on a web browser and update your country setting in Profile.",
                "3. Note: Premium accounts do not have the 14-day travel restriction and work worldwide."
            ],
            "source_tweet_sample": "@484611 Free accounts have a 14-day limit abroad. You can update your country settings on your account page."
        },
        {
            "id": "AUTH_03_OFFLINE_LOCKOUT",
            "name": "Offline Mode Authentication Renewal",
            "triggers": ["logged out in offline mode", "can't log in offline", "offline mode lockout"],
            "steps": [
                "1. Connect your device to an active Wi-Fi or cellular network.",
                "2. Launch the app and log in while connected to refresh authentication security tokens.",
                "3. Once logged in, you can safely re-enable Offline Mode in Settings > Playback."
            ],
            "source_tweet_sample": "@355584 To log back in, your device needs an active internet connection to authenticate your credentials."
        }
    ],

    "device_platform": [
        {
            "id": "DEV_01_POWER_CYCLE_CONNECT",
            "name": "Console & Smart TV Power Cycle / Spotify Connect",
            "triggers": ["ps4", "ps5", "playstation", "xbox", "tv", "smart tv", "spotify connect", "roku"],
            "steps": [
                "1. Completely power down the console or TV for 30–60 seconds (unplug power cable from wall).",
                "2. Verify that your phone/computer and your console/TV are connected to the exact same Wi-Fi network name/band.",
                "3. Open Spotify on your mobile app and select Devices Available (Spotify Connect) to pair."
            ],
            "source_tweet_sample": "@116387 Can you try power cycling your console by unplugging it for 30 seconds? Keep us posted."
        },
        {
            "id": "DEV_02_BLUETOOTH_PAIRING",
            "name": "Bluetooth & Speaker Re-pairing Protocol",
            "triggers": ["bluetooth", "speaker", "car bluetooth", "headphones", "audio disconnect"],
            "steps": [
                "1. In your device Settings > Bluetooth, 'Forget' or remove the speaker/headphones.",
                "2. Turn off Bluetooth on your device, restart the device, and turn Bluetooth back on.",
                "3. Put your speaker/headphones into pairing mode and re-pair."
            ],
            "source_tweet_sample": "@116887 Could you try forgetting the Bluetooth device, restarting your phone, and pairing them again?"
        },
        {
            "id": "DEV_03_CAR_INTEGRATION",
            "name": "CarPlay & Android Auto Connection Refresh",
            "triggers": ["carplay", "android auto", "car audio", "car display", "head unit"],
            "steps": [
                "1. Disconnect and re-connect the USB cable (or re-pair Bluetooth connection).",
                "2. Check phone settings to ensure CarPlay / Android Auto permissions are enabled for Spotify.",
                "3. Restart both the phone and your car's infotainment head unit."
            ],
            "source_tweet_sample": "@159354 We recommend checking device permissions for your car integration and restarting the head unit."
        }
    ],

    "premium_billing": [
        {
            "id": "BILL_01_FAMILY_INVITE",
            "name": "Family & Duo Plan Invitation / Address Match",
            "triggers": ["family plan invite", "can't add family member", "error 3", "invite link", "duo plan"],
            "steps": [
                "1. The plan manager must generate a new invite link from their account page.",
                "2. The invited family member should open the link in a Private/Incognito browser window.",
                "3. Crucial: The invitee must enter the exact same residential home address as the plan manager."
            ],
            "source_tweet_sample": "@141626 The T&Cs were set in place to make sure only your family members living at the same address enjoy the Premium features."
        },
        {
            "id": "BILL_02_STUDENT_DISCOUNT",
            "name": "Student Discount Verification & Renewal",
            "triggers": ["student discount", "sheerid", "student verification", "college discount"],
            "steps": [
                "1. Log into the student discount portal via the Spotify website on a web browser.",
                "2. Complete student enrollment verification through SheerID.",
                "3. Note: Student discount must be renewed every 12 months (up to a maximum of 4 years)."
            ],
            "source_tweet_sample": "@294688 You can sign up for or re-verify your student discount by heading to our student verification page."
        },
        {
            "id": "BILL_03_CANCELLATION_DOWNGRADE",
            "name": "Subscription Cancellation & Cycle Expiration",
            "triggers": ["cancel subscription", "stop premium", "how to cancel", "downgrade to free"],
            "steps": [
                "1. Log into your account page on a web browser and select 'Change plan' > 'Cancel Premium'.",
                "2. Note: Your Premium features will remain active until your current paid billing cycle ends.",
                "3. After the cycle ends, your account automatically reverts to Free with all playlists preserved."
            ],
            "source_tweet_sample": "@141285 Canceling the subscription can be done anytime. You'll still keep Premium until your next billing date."
        }
    ]
}

class CuratedPolicyReference:
    """Named grounding sub-component for the Phase 4 Reply Drafter."""

    def __init__(self):
        self.policies = TROUBLESHOOTING_POLICIES
        self.total_sops = sum(len(sops) for sops in self.policies.values())

    def get_policies_for_intent(self, intent: str) -> List[Dict[str, Any]]:
        """Retrieve all standard operating procedures for an intent."""
        return self.policies.get(intent, [])

    def find_matching_sop(self, intent: str, query_text: str) -> Optional[Dict[str, Any]]:
        """Find the most specific SOP matching customer query keywords."""
        sops = self.get_policies_for_intent(intent)
        if query_text is None:
            return sops[0] if sops else None
        q_lower = str(query_text).strip().lower()
        if not q_lower:
            return sops[0] if sops else None
        
        # Check for specific trigger keyword matches
        for sop in sops:
            for trigger in sop["triggers"]:
                if trigger in q_lower:
                    return sop
        
        # Fallback to the primary general SOP (first in list)
        return sops[0] if sops else None

    def format_policy_context(self, intent: str, query_text: str) -> str:
        """Format grounded policy guidance to inject into LLM drafting prompt."""
        sop = self.find_matching_sop(intent, query_text)
        if not sop:
            return "General Spotify Support Guidance: Recommend checking account status and restarting device."

        formatted_steps = "\n".join(sop["steps"])
        return (
            f"Grounding Policy Reference: [{sop['name']}]\n"
            f"Applicable Procedure:\n{formatted_steps}"
        )

if __name__ == "__main__":
    ref = CuratedPolicyReference()
    print(f"Loaded CuratedPolicyReference with {ref.total_sops} SOPs across {len(ref.policies)} intents (Zero synthetic URLs).")
    for intent, sops in ref.policies.items():
        print(f"  - {intent}: {len(sops)} SOPs")
