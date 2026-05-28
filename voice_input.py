"""
voice_input.py - uses CSS transform to reposition without breaking click target
"""

import streamlit as st
from groq import Groq
import hashlib


def voice_input_component() -> str | None:

    st.markdown("""
        <style>
        /* Container that wraps stAudioInput — move this to bottom bar area */
        div[data-testid="stElementContainer"]:has(div[data-testid="stAudioInput"]) {
            position: fixed !important;
            bottom:93px !important;
            right: 210px !important;
            z-index: 99999 !important;
            width: 42px !important;
            height: 42px !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        div[data-testid="stAudioInput"] {
            width: 42px !important;
            height: 42px !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
            border: none !important;
        }

        div[data-testid="stAudioInput"] label {
            display: none !important;
        }

        /* Outer gold circle */
        div[data-testid="stAudioInput"] > div {
            width: 42px !important;
            height: 42px !important;
            border-radius: 50% !important;
            background: rgba(10,8,2,0.97) !important;
            border: 1.5px solid rgba(212,175,55,0.55) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            overflow: hidden !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            padding: 0 !important;
            position: relative !important;
        }

        div[data-testid="stAudioInput"] > div:hover {
            border-color: #d4af37 !important;
            box-shadow: 0 0 14px rgba(212,175,55,0.3) !important;
        }

        /* Hide waveform/timer — but NOT the button or its ancestors */
div[data-testid="stAudioInput"] > div > *:not(button):not(:has(button)) {
    display: none !important;
}

/* Target button anywhere inside, not just direct child */
div[data-testid="stAudioInput"] button {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    width: 42px !important;
    height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    pointer-events: auto !important;
}

div[data-testid="stAudioInput"] button svg {
    width: 17px !important;
    height: 17px !important;
    stroke: rgba(212,175,55,0.9) !important;
    fill: none !important;
}

        /* Mic button */
        div[data-testid="stAudioInput"] > div > button {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            width: 42px !important;
            height: 42px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            outline: none !important;
        }

        /* Gold mic SVG */
        div[data-testid="stAudioInput"] > div > button svg {
            width: 17px !important;
            height: 17px !important;
            stroke: rgba(212,175,55,0.9) !important;
            fill: none !important;
        }

        /* Pulse when recording */
        @keyframes mic-ring {
            0%   { box-shadow: 0 0 0 0px rgba(212,175,55,0.5); }
            70%  { box-shadow: 0 0 0 9px rgba(212,175,55,0);   }
            100% { box-shadow: 0 0 0 0px rgba(212,175,55,0);   }
        }
        /* Hide Streamlit's default record-square SVG */
div[data-testid="stAudioInput"] button svg {
    display: none !important;
}

/* Inject a real mic icon using SVG data URI */
div[data-testid="stAudioInput"] button::before {
    content: "";
    display: block;
    width: 18px;
    height: 18px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23d4af37' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='2' width='6' height='12' rx='3'/%3E%3Cpath d='M5 10a7 7 0 0 0 14 0'/%3E%3Cline x1='12' y1='19' x2='12' y2='22'/%3E%3Cline x1='9' y1='22' x2='15' y2='22'/%3E%3C/svg%3E") center/contain no-repeat;
    pointer-events: none;
}
/* ── Outer wrapper ── */
div[data-testid="stElementContainer"]:has(div[data-testid="stAudioInput"]) {
    position: fixed !important;
    bottom: 93px !important;
    right: 210px !important;
    z-index: 99999 !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    margin: 0 !important;
    pointer-events: auto !important;
}

div[data-testid="stAudioInput"] {
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    overflow: hidden !important;
}

/* Hide label */
div[data-testid="stAudioInput"] label {
    display: none !important;
}

/* Gold circle — locked size always */
div[data-testid="stAudioInput"] > div {
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    border-radius: 50% !important;
    background: rgba(10,8,2,0.97) !important;
    border: 1.5px solid rgba(212,175,55,0.55) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    cursor: pointer !important;
    padding: 0 !important;
    flex-direction: row !important;
}

/* Kill waveform, timer, progress bar — everything except the button */
div[data-testid="stAudioInput"] > div > *:not(button):not(:has(button)) {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    flex: none !important;
}

/* Button — locked size */
div[data-testid="stAudioInput"] button {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    flex: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    position: relative !important;
}

/* Hide Streamlit's SVG */
div[data-testid="stAudioInput"] button svg {
    display: none !important;
}

/* Mic icon — idle */
div[data-testid="stAudioInput"] button::before {
    content: "";
    display: block;
    width: 18px;
    height: 18px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23d4af37' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='2' width='6' height='12' rx='3'/%3E%3Cpath d='M5 10a7 7 0 0 0 14 0'/%3E%3Cline x1='12' y1='19' x2='12' y2='22'/%3E%3Cline x1='9' y1='22' x2='15' y2='22'/%3E%3C/svg%3E") center/contain no-repeat;
    pointer-events: none;
    transition: opacity 0.2s ease;
}

/* Pulse ring when recording (aria-label changes to "Stop recording") */
div[data-testid="stAudioInput"] button[aria-label*="Stop"]::after {
    content: "";
    position: absolute;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    animation: mic-ring 1.2s ease-out infinite;
    pointer-events: none;
}

/* Mic turns brighter red when recording */
div[data-testid="stAudioInput"] button[aria-label*="Stop"]::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ff4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='2' width='6' height='12' rx='3'/%3E%3Cpath d='M5 10a7 7 0 0 0 14 0'/%3E%3Cline x1='12' y1='19' x2='12' y2='22'/%3E%3Cline x1='9' y1='22' x2='15' y2='22'/%3E%3C/svg%3E");
}

@keyframes mic-ring {
    0%   { box-shadow: 0 0 0 0px rgba(212,175,55,0.5); }
    70%  { box-shadow: 0 0 0 9px rgba(212,175,55,0.0); }
    100% { box-shadow: 0 0 0 0px rgba(212,175,55,0.0); }
}
/* The audio player element */
div[data-testid="stAudioInput"] audio {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* Playback controls bar */
div[data-testid="stAudioInput"] [class*="audioControls"],
div[data-testid="stAudioInput"] [class*="playback"],
div[data-testid="stAudioInput"] [class*="waveform"],
div[data-testid="stAudioInput"] [class*="duration"],
div[data-testid="stAudioInput"] [class*="download"],
div[data-testid="stAudioInput"] [class*="timer"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    flex: none !important;
}

/* Nuclear option — hide ALL direct children of the inner div
   except the button and its ancestors */
div[data-testid="stAudioInput"] > div > *:not(button):not(:has(button)) {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    flex: none !important;
    position: absolute !important;  /* ← pulls it out of flex flow entirely */
}

/* Re-lock the circle size after playback state */
div[data-testid="stAudioInput"] > div {
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    overflow: hidden !important;
    flex-wrap: nowrap !important;  /* ← prevents wrapping to second row */
}
/* Button — absolute so flex siblings can NEVER push it */
div[data-testid="stAudioInput"] button {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    flex: none !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    z-index: 2 !important;
}

/* Make the circle a positioning context */
div[data-testid="stAudioInput"] > div {
    position: relative !important;   /* ← ADD THIS */
    width: 42px !important;
    height: 42px !important;
    min-width: 42px !important;
    max-width: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    border-radius: 50% !important;
    background: rgba(10,8,2,0.97) !important;
    border: 1.5px solid rgba(212,175,55,0.55) !important;
    overflow: hidden !important;
    padding: 0 !important;
}

/* Nuke ALL siblings — doesn't matter what Streamlit injects */
div[data-testid="stAudioInput"] > div > *:not(button):not(:has(button)) {
    display: none !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    z-index: -1 !important;
}
        </style>
    """, unsafe_allow_html=True)

    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None

    audio = st.audio_input(
        label="mic",
        key="cricket_audio_input",
        label_visibility="collapsed",
    )

    if audio is None:
        return None

    audio_bytes = audio.read()
    audio_hash  = hashlib.md5(audio_bytes).hexdigest()

    if audio_hash == st.session_state.last_audio_hash:
        return None

    st.session_state.last_audio_hash = audio_hash

    try:
        api_key = st.secrets["groq"]["key"]
    except KeyError:
        st.warning("⚠️ Groq key missing in secrets.toml")
        return None

    with st.spinner("Transcribing…"):
        try:
            client = Groq(api_key=api_key)
            result = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=("audio.wav", audio_bytes, "audio/wav"),
                language="en",
            )
            text = result.text.strip()
            return text if text else None
        except Exception as e:
            st.warning(f"⚠️ Transcription failed: {str(e)}")
            return None