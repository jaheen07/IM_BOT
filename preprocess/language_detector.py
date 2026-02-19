from langdetect import detect


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "bn":
            return "bangla"
        elif lang == "en":
            return "english"
        else:
            return "english"  # fallback
    except Exception:
        return "english"  # fallback
