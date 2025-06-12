import os
import csv

VALIDATION_OUTPUT_DIR = "C:/Users/Iris/Documents/RI_Training_Data/ComparisonOutput"
summary_csv = os.path.join(VALIDATION_OUTPUT_DIR, "average_similarities_all_epochs.csv")

epochs = 40  # Adjust this if needed

with open(summary_csv, mode='w', newline='') as summary_file:
    writer = csv.writer(summary_file)
    writer.writerow(["Epoch", "Average Similarity"])

    for epoch in range(1, epochs + 1):
        sim_file = os.path.join(VALIDATION_OUTPUT_DIR, f"epoch_{epoch}", f"similarities_epoch_{epoch}.csv")
        if not os.path.exists(sim_file):
            print(f"File not found for epoch {epoch}: {sim_file}")
            continue

        with open(sim_file, mode='r') as f:
            reader = csv.DictReader(f)
            sims = [float(row["Similarity"]) for row in reader if row["Similarity"] != ""]
            avg_sim = sum(sims) / len(sims) if sims else 0.0

        writer.writerow([epoch, avg_sim])
        print(f"Epoch {epoch} - Average Similarity: {avg_sim:.4f}")

print(f"Summary CSV saved to {summary_csv}")
