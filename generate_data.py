import pandas as pd
import random

def generate_data():
    train_s1 = [
        ["S1-001", "Sharma Traders Pvt. Ltd.", "12 MG Road, Bangalore", "India"],
        ["S1-002", "Acme Corporation", "123 Main St, New York, NY", "US"],
        ["S1-003", "Tech Solutions Ltd", "45 Park Avenue, London", "UK"],
        ["S1-004", "Global Imports", "78 Ocean Drive, Miami, FL", "US"],
        ["S1-005", "No Match Company", "Somewhere", "India"] # singleton
    ]
    train_s2 = [
        ["S2-001", "Sharma Tradres Private Limited", "12 M.G. Rd", "India"],
        ["S2-002", "Acme Corp.", "123 Main Street, NYC", "US"],
        ["S2-003", "Global Imports Inc", "Ocean Dr 78, Miami", "US"],
        ["S2-004", "Tech Sol Ltd", "45 Park Ave, London", "UK"]
    ]
    train_s3 = [
        ["S3-001", "Sharma Traders", "Near SBI ATM, MG Road", "India"],
        ["S3-002", "Acme", "123 Main", "US"],
        ["S3-003", "Tech Solutions", "Park Avenue 45", "UK"],
        ["S3-004", "Random Company", "Unknown", "US"]
    ]
    train_gt = [
        ["S1-001", "S2-001,S3-001"],
        ["S1-002", "S2-002,S3-002"],
        ["S1-003", "S2-004,S3-003"],
        ["S1-004", "S2-003"],
        ["S1-005", ""]
    ]

    test_s1 = [
        ["S1-101", "Boulangerie Paris", "10 Rue de la Paix, Paris", "France"],
        ["S1-102", "Berlin Auto GmbH", "5 Unter den Linden, Berlin", "Germany"],
        ["S1-103", "Tokyo Electronics", "Shibuya Crossing", "Japan"],
        ["S1-104", "Rio Coffee", "Copacabana Beach", "Brazil"],
        ["S1-105", "Lonely Shop", "Lonely St", "France"]
    ]
    test_s2 = [
        ["S2-101", "Boulangerie Paris", "Rue de la Paix 10", "France"],
        ["S2-102", "Berlin Automotives", "Unter den Linden 5", "Germany"],
        ["S2-103", "Tokyo Elec", "Shibuya", "Japan"]
    ]
    test_s3 = [
        ["S3-101", "Paris Boulangerie", "10 Rue de la Paix", "France"],
        ["S3-102", "Berlin Auto", "Berlin", "Germany"],
        ["S3-103", "Rio Coffee Corp", "Copacabana", "Brazil"],
        ["S3-104", "Noise Co", "Noise St", "Germany"]
    ]

    cols_s = ["entity_id", "business_name", "business_address", "country"]
    cols_gt = ["entity_id", "match_ids"]

    pd.DataFrame(train_s1, columns=cols_s).to_csv("data/train/train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(train_s2, columns=cols_s).to_csv("data/train/train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(train_s3, columns=cols_s).to_csv("data/train/train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(train_gt, columns=cols_gt).to_csv("data/train/train_ground_truth.tsv", sep="\t", index=False)

    pd.DataFrame(test_s1, columns=cols_s).to_csv("data/test/test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(test_s2, columns=cols_s).to_csv("data/test/test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(test_s3, columns=cols_s).to_csv("data/test/test_source3.tsv", sep="\t", index=False)

if __name__ == "__main__":
    generate_data()
