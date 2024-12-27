import json

def write_dict_to_file(file_path: str = None, data_dict: dict = None):
    """
    Write a dictionary to a file as a JSON object.
    """
    with open(file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)

# def write_test_cases_to_file(test_cases, output_path):
#     """
#     Write a dictionary of test cases to a file.
#     """
#     with open(output_path, 'w', encoding='utf-8') as f:
#         # Write header
#         f.write("TEST CASES\n")
#         f.write("=" * 50 + "\n\n")
        
#         for outer_key, inner_dict in test_cases.items():
#             for category, content in inner_dict.items():
#                 # Format category title
#                 category_title = category.upper().replace('_', ' ')
#                 f.write(f"\n{category_title}\n")
#                 f.write("-" * len(category_title) + "\n\n")
                
#                 # Write the test cases content
#                 if isinstance(content, str):
#                     f.write(content.strip())
#                     f.write("\n\n" + "=" * 50 + "\n")