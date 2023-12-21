import csv

def remove_duplicate_faqs(input_file_path, output_file_path):
    unique_faqs = {}
    
    with open(input_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['question'].strip()
            if question not in unique_faqs:
                unique_faqs[question] = row['answer'].strip()

    with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['question', 'answer'])
        writer.writeheader()
        for question, answer in unique_faqs.items():
            writer.writerow({'question': question, 'answer': answer})

    print(f"Processed {len(unique_faqs)} unique FAQs.")

# File paths
input_file_path = 'faq.csv'
output_file_path = 'unique_faqs.csv'

# Process the file to remove duplicates
remove_duplicate_faqs(input_file_path, output_file_path)
