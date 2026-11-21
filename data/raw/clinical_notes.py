
import json
import random
import re
from pathlib import Path
from faker import Faker

fake = Faker("en_IN")

random.seed(42)
Faker.seed(42)

TOTAL_RECORDS = 1000

OUTPUT_PATH = Path(
    "data/raw/synthetic_clinical_notes.json"
)

EMAIL_DOMAINS = [
    "example.com",
    "example.org",
    "example.net",
    "example.co.in",
    "mail.example.com",
    "hospital.example.org",
    "clinic.example.net",
    "health.example.com",
    "care.example.org",
    "patient.example.net",
]

CONDITIONS = [
    "Type 2 diabetes",
    "Hypertension",
    "Asthma",
    "Migraine",
    "Common cold",
    "Viral fever",
    "Back pain",
    "Cardiac assessment",
    "Respiratory infection",
    "High blood pressure",
]

TEMPLATES = [
    (
        "Patient {name}, aged {age}, visited the hospital. "
        "Contact: {phone}. Email: {email}. "
        "Medical record number: {mrn}. "
        "Diagnosis: {condition}."
    ),
    (
        "{name} visited the outpatient clinic at age {age}. "
        "The patient can be reached at {phone}. "
        "Email address: {email}. "
        "Medical record ID: {mrn}. "
        "Reported condition: {condition}."
    ),
    (
        "A new patient, {name} (age {age}), was registered today. "
        "Phone number: {phone}. "
        "Send updates to {email}. "
        "Patient identifier: {mrn}. "
        "Clinical impression: {condition}."
    ),
    (
        "During the consultation, {name} provided contact details: "
        "{phone} and {email}. "
        "The patient is {age} years old. "
        "Hospital record number: {mrn}. "
        "Reason for visit: {condition}."
    ),
    (
        "Please arrange a follow-up for {name}. "
        "Contact the patient on {phone} or write to {email}. "
        "The file reference is {mrn}. "
        "Age: {age}. Current diagnosis: {condition}."
    ),
    (
        "The clinic received a referral for {name}, age {age}. "
        "Referral contact: {phone}. "
        "Patient email: {email}. "
        "Reference code: {mrn}. "
        "Referral reason: {condition}."
    ),
    (
        "Registration details: patient {name}; age {age}; "
        "telephone {phone}; electronic mail {email}; "
        "medical identifier {mrn}. "
        "Assessment: {condition}."
    ),
    (
        "At the recent visit, {name} reported {condition}. "
        "Staff recorded phone {phone}, email {email}, "
        "and chart number {mrn}. "
        "Patient age: {age}."
    ),
]


def generate_name(used_names):
    """Generate a unique two-part patient name."""

    while True:
        first = re.sub(
            r"[^A-Za-zÀ-ÖØ-öø-ÿ]",
            "",
            fake.first_name(),
        ).title()

        last = re.sub(
            r"[^A-Za-zÀ-ÖØ-öø-ÿ]",
            "",
            fake.last_name(),
        ).title()

        name = f"{first} {last}"

        if (
            first
            and last
            and name.casefold() not in used_names
        ):
            used_names.add(name.casefold())
            return name


def generate_phone(index, used_phones):
    """Generate supported Indian phone number formats."""

    while True:
        number = str(
            random.randint(6000000000, 9999999999)
        )

        if number not in used_phones:
            used_phones.add(number)
            break

    formats = [
        f"+91 {number}",
        f"+91-{number}",
        f"0{number}",
        number,
    ]

    return formats[index % len(formats)]


def generate_email(name, index):
    """Generate diverse, unique synthetic email addresses."""

    first, last = name.lower().split()

    first = re.sub(r"[^a-z0-9]", "", first)
    last = re.sub(r"[^a-z0-9]", "", last)

    patterns = [
        f"{first}.{last}{index}",
        f"{first}{last}{index}",
        f"{first[0]}{last}{index}",
        f"{last}_{first}{index}",
        f"{first}{index}.{last}",
    ]

    username = patterns[index % len(patterns)]
    domain = EMAIL_DOMAINS[index % len(EMAIL_DOMAINS)]

    return f"{username}@{domain}"


def generate_mrn(index, used_mrns):
    """Generate unique medical record numbers."""

    while True:
        number = random.randint(100000, 999999)

        formats = [
            f"MRN{number}",
            f"MRN-{number}",
            f"MRN {number}",
        ]

        mrn = formats[index % len(formats)]

        if mrn not in used_mrns:
            used_mrns.add(mrn)
            return mrn


def generate_clinical_note(index, used_names, used_phones, used_mrns):

    name = generate_name(used_names)
    phone = generate_phone(index, used_phones)
    email = generate_email(name, index)
    mrn = generate_mrn(index, used_mrns)

    age = random.randint(18, 85)
    condition = random.choice(CONDITIONS)

    template = TEMPLATES[index % len(TEMPLATES)]

    note = template.format(
        name=name,
        age=age,
        phone=phone,
        email=email,
        mrn=mrn,
        condition=condition,
    )

    return {
        "text": note,
        "entities": {
            "PERSON": name,
            "PHONE": phone,
            "EMAIL": email,
            "MRN": mrn,
        },
    }


def validate_dataset():

    from src.deidentifier import deidentify

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    labels = ["PERSON", "PHONE", "EMAIL", "MRN"]

    counts = {label: 0 for label in labels}
    missed_examples = {label: [] for label in labels}

    for item in dataset:

        _, mapping = deidentify(item["text"])

        for label in labels:

            expected = item["entities"][label]

            detected_values = [
                value
                for key, value in mapping.items()
                if key.startswith(f"[{label}_")
            ]

            if expected in detected_values:
                counts[label] += 1
            else:
                missed_examples[label].append(expected)

    print("\n========== DATASET VALIDATION ==========")
    print("Total notes:", len(dataset))

    all_passed = True

    for label, count in counts.items():

        print(f"{label}: {count}/{len(dataset)}")

        if count != len(dataset):
            all_passed = False

    print("\n========== MISSED PHI EXAMPLES ==========")

    for label, examples in missed_examples.items():

        print(f"\n{label} missed: {len(examples)}")

        for value in examples[:5]:
            print("  -", value)

    if all_passed:
        print("\nALL PHI CATEGORIES PASSED!")
    else:
        print("\nWARNING: SOME PHI ENTITIES WERE MISSED.")

    return all_passed


def main():

    used_names = set()
    used_phones = set()
    used_mrns = set()

    dataset = [
        generate_clinical_note(
            index,
            used_names,
            used_phones,
            used_mrns,
        )
        for index in range(TOTAL_RECORDS)
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n========== SYNTHETIC DATASET GENERATED ==========")
    print("Total clinical notes:", len(dataset))
    print("Unique patient names:", len(used_names))
    print("Unique phone numbers:", len(used_phones))
    print("Unique MRNs:", len(used_mrns))
    print("Email domains:", len(EMAIL_DOMAINS))
    print("Clinical templates:", len(TEMPLATES))
    print("Saved to:", OUTPUT_PATH)

    print("\nSample clinical note:\n")
    print(dataset[0]["text"])

    validate_dataset()


if __name__ == "__main__":
    main()
