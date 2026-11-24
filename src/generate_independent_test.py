import json
from pathlib import Path

OUTPUT_PATH = Path("data/raw/synthetic_clinical_notes_test.json")

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun",
    "Reyansh", "Ayaan", "Krishna", "Ishaan", "Shaurya",
    "Atharv", "Dhruv", "Kabir", "Ritvik", "Anirudh",
    "Diya", "Ananya", "Aadhya", "Saanvi", "Pari",
    "Myra", "Ira", "Aarohi", "Anika", "Navya",
    "Meera", "Tara", "Kiara", "Riya", "Ishita",
    "Kavya", "Nisha", "Tanvi", "Pooja", "Shruti",
    "Dev", "Karthik", "Nikhil", "Rahul", "Siddharth",
    "Varun", "Rohan", "Manav", "Pranav", "Harish",
    "Gauri", "Lavanya", "Madhuri", "Sneha", "Trisha",
]

LAST_NAMES = [
    "Sharma", "Iyer", "Reddy", "Nair", "Mehta",
    "Kapoor", "Menon", "Pillai", "Joshi", "Patel",
    "Rao", "Verma", "Das", "Kulkarni", "Bose",
    "Malhotra", "Desai", "Mishra", "Chawla", "Sinha",
    "Bhat", "Shetty", "Ghosh", "Chandra", "Khanna",
    "Agarwal", "Naidu", "Thomas", "Fernandes", "Dutta",
    "Saxena", "Prasad", "Jain", "Mohan", "Gupta",
    "Subramanian", "Krishnan", "Mukherjee", "Banerjee",
    "Chatterjee", "Menon", "Hegde", "Pawar", "Sethi",
    "Kumar", "Singh", "Chopra", "Rana", "Roy", "Paul",
]

CONDITIONS = [
    "acute bronchitis", "viral gastroenteritis",
    "a mild respiratory infection", "dehydration",
    "seasonal allergic rhinitis", "a urinary tract infection",
    "migraine", "muscle strain", "hypertension review",
    "type two diabetes review",
]

def build_record(index, template_id):
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index * 7) % len(LAST_NAMES)]
    name = f"{first} {chr(65 + index // 50)}. {last}"

    digits = f"{8000000000 + index:010d}"
    phone = f"+91 {digits[:3]} {digits[3:6]} {digits[6:]}"
    email = f"{first.lower()}.{last.lower()}{index}@testmail.example"
    mrn = f"MRN-{83000000 + index:08d}"
    condition = CONDITIONS[index % len(CONDITIONS)]

    if template_id == 8:
        text = (
            f"DISCHARGE SUMMARY\n"
            f"Following inpatient treatment, {name} was discharged "
            f"in stable condition. The admission was related to "
            f"{condition}. For post-discharge coordination, contact "
            f"the patient at {phone} or send correspondence to "
            f"{email}. The hospital file reference is {mrn}. "
            f"Follow-up was advised within two weeks."
        )

    elif template_id == 9:
        text = (
            f"EMERGENCY DEPARTMENT RECORD\n"
            f"Arrival details were documented for {name}. "
            f"The attending team assessed {condition}. "
            f"Next-of-kin communication details supplied by the "
            f"patient were {phone} and {email}. "
            f"Encounter identification: {mrn}. "
            f"The patient was discharged with routine instructions."
        )

    elif template_id == 10:
        text = (
            f"REFERRAL CORRESPONDENCE\n"
            f"Dear Specialist, please review {name} for "
            f"{condition}. The referring clinic can coordinate "
            f"appointments through the patient contact number "
            f"{phone}. Patient correspondence address: {email}. "
            f"Please quote reference {mrn} in your response."
        )

    else:
        text = (
            f"FOLLOW-UP CLINIC ENTRY\n"
            f"At today's review, {name} discussed ongoing care "
            f"for {condition}. The clinic verified the telephone "
            f"contact as {phone} and the electronic contact as "
            f"{email}. The longitudinal medical file is recorded "
            f"under {mrn}. A further review was recommended."
        )

    return {
        "text": text,
        "entities": {
            "PERSON": name,
            "PHONE": phone,
            "EMAIL": email,
            "MRN": mrn,
        },
        "template_id": template_id,
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Inspect it before deciding whether to replace it."
        )

    records = []

    for template_id in range(8, 12):
        for offset in range(50):
            index = (template_id - 8) * 50 + offset
            records.append(build_record(index, template_id))

    # Validate record count, template distribution and annotations.
    assert len(records) == 200
    assert {
        template_id: sum(
            record["template_id"] == template_id
            for record in records
        )
        for template_id in range(8, 12)
    } == {8: 50, 9: 50, 10: 50, 11: 50}

    for record in records:
        text = record["text"]
        for label, value in record["entities"].items():
            assert value in text, (label, value)
            assert text.count(value) == 1, (label, value)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print("\n========== INDEPENDENT TEST DATASET ==========")
    print("Records:", len(records))
    print("Template distribution: 8–11, 50 each")
    print("Entity annotations: 4 per record")
    print("Annotation validation: passed")
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
