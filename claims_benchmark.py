"""
claims_benchmark.py - balance the benchmark with known-FALSE and mixed-evidence claims.
Runs them through the same search + verdict pipeline; gold_label is pre-filled.

Usage:  python claims_benchmark.py results_synthetic.xlsx
Copy its rows into your main results sheet (or report both).
"""
import sys
import time

from factcheck import pipeline
from factcheck.models import Claim
from benchmark import rows_for, write_book

FALSE_CLAIMS = [
    ("The human esophagus is about 60 centimeters long.", "Science"),
    ("Edward Jenner gave the first smallpox vaccine in 1850.", "History"),
    ("The human heart has six chambers.", "Science"),
    ("The Great Wall of China is visible to the naked eye from the Moon.", "Science"),
    ("Lightning never strikes the same place twice.", "Science"),
    ("Humans use only 10 percent of their brains.", "Science"),
    ("The Sun orbits the Earth once a year.", "Science"),
    ("Vaccines cause autism.", "Health"),
    ("Water boils at 50 degrees Celsius at sea level.", "Science"),
    ("The Earth is only about 6,000 years old.", "Science"),
    ("Mount Everest is located in Brazil.", "General"),
    ("Antibiotics are effective against viral infections such as the common cold.", "Health"),
    ("An adult human has 306 bones.", "Science"),
    ("The Pacific Ocean is the smallest ocean on Earth.", "General"),
    ("Vitamin C cures cancer.", "Health"),
]

MIXED_CLAIMS = [  # contested / mixed evidence -> gold UNVERIFIABLE (merged class)
    ("Coffee is bad for your health.", "Health"),
    ("Eating eggs causes heart disease.", "Health"),
    ("Organic food is healthier than conventional food.", "Health"),
    ("Red wine is good for your heart.", "Health"),
    ("Cell phones cause brain cancer.", "Health"),
    ("Artificial sweeteners cause cancer in humans.", "Health"),
    ("Playing video games makes children more violent.", "Science"),
    ("Humans will colonize Mars by 2040.", "Technology"),
    ("Daily multivitamins improve the health of healthy adults.", "Health"),
    ("Fasting is the best way to lose weight.", "Health"),
]

ITEMS = [(t, c, "FALSE") for t, c in FALSE_CLAIMS] + [(t, c, "UNVERIFIABLE") for t, c in MIXED_CLAIMS]
CHUNK = 5  # claims per round, to stay under the Groq tokens-per-minute limit


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results_synthetic.xlsx"
    gold = {t: g for t, _, g in ITEMS}
    all_rows = []

    for start in range(0, len(ITEMS), CHUNK):
        part = ITEMS[start:start + CHUNK]
        claims = [Claim(claim_id=i, claim_text=t, start_time="00:00", end_time="00:00",
                        category=c, importance="medium", search_queries=[])
                  for i, (t, c, _) in enumerate(part, 1)]
        for attempt in range(1, 6):
            try:
                results = pipeline.get_claim_verdicts(claims, True)
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    print(f"rate limit, waiting 65s (attempt {attempt}/5)")
                    time.sleep(65)
                else:
                    raise
        else:
            continue
        for row in rows_for("synthetic", claims, results):
            row[1] = len(all_rows) + 1
            row[14] = gold[row[2]]
            all_rows.append(row)
        write_book(all_rows, out)
        print(f"{len(all_rows)}/{len(ITEMS)} done -> {out}")
        time.sleep(60)
    print("Done.")


if __name__ == "__main__":
    main()