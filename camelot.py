"""Table de conversion note + mode → notation Camelot wheel."""

CAMELOT = {
    ("C",  "major"): "8B",  ("A",  "minor"): "8A",
    ("Db", "major"): "3B",  ("Bb", "minor"): "3A",
    ("D",  "major"): "10B", ("B",  "minor"): "10A",
    ("Eb", "major"): "5B",  ("C",  "minor"): "5A",
    ("E",  "major"): "12B", ("Db", "minor"): "12A",
    ("F",  "major"): "7B",  ("D",  "minor"): "7A",
    ("F#", "major"): "2B",  ("Eb", "minor"): "2A",
    ("G",  "major"): "9B",  ("E",  "minor"): "9A",
    ("Ab", "major"): "4B",  ("F",  "minor"): "4A",
    ("A",  "major"): "11B", ("F#", "minor"): "11A",
    ("Bb", "major"): "6B",  ("G",  "minor"): "6A",
    ("B",  "major"): "1B",  ("Ab", "minor"): "1A",
}

def to_camelot(key: str, scale: str) -> str:
    return CAMELOT.get((key, scale), "?")
