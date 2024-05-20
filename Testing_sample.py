import json

questions = [
    "1. How do age and gender demographics correlate with the preferences for specific categories like \"Economic Policies\", \"Healthcare\", and \"Gun Control\"?",
    "2. What are the most positively and negatively perceived categories across different age groups?",
    "3. Are there any notable differences in the mean satisfaction scores between males and females?",
    "4. Which topics are most frequently mentioned in positive, neutral, and negative categories by consumers?",
    "5. How do the principal component scores (PC1 and PC2) relate to the clustering of consumer responses?",
    "6. Can we identify any outliers or unusual trends in consumer feedback based on the clustering results?"
  ]

response_json = json.dumps({"questions": [question.strip() for question in questions]})
print(response_json)
