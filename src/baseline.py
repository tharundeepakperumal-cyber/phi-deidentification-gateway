from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

def detect_phi(text):
    results = analyzer.analyze(
        text=text,
        language="en"
    )

    for item in results:
        print(
            text[item.start:item.end],
            item.entity_type,
            item.score
        )

if __name__ == "__main__":
    note = """
    Patient John Smith visited Chennai Hospital.
    Phone: 9876543210
    """

    detect_phi(note)