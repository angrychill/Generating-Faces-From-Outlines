from deepface import DeepFace

def compare_faces(original_image_path, generated_image_path):
    # Compare the two images and return the similarity score
    result = DeepFace.verify(original_image_path, generated_image_path)
    
    # Result contains the similarity score and if the images belong to the same person
    return result

# Example usage
original_image_path = "PATH/VALID_DATA/Naomi_Watts_0001.jpg"
generated_image_path = "PATH/PROCESSED_VALID_DATA/Naomi_Watts_0001.jpg"
result = compare_faces(original_image_path, generated_image_path)

print("Are the faces the same?", result["verified"])
print("Similarity score:", result["distance"])

"""
Score ~0.0 to ~0.3: High similarity (likely same person).
Score ~0.3 to ~0.5: Moderate similarity (could be same person, but may need further inspection).
Score ~0.5 to ~0.8: Low similarity (likely different people).
Score > 0.8: Very low similarity (definitely different people).

Best score (same person): 0.0
Worst score (completely different people): 1.0
"""