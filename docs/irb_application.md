# IRB Application: RajNLP-50K Annotation Study

**Protocol Title**: Construction of a Rajasthani-Hindi Code-Switched NLP Corpus through Human Annotation  
**Principal Investigator**: [Your Name], [Your Department], [Your Institution]  
**Co-Investigators**: [Names if any]  
**Submission Date**: [Date]  
**Expected Start Date**: [Date]  
**Expected End Date**: [Date]  

---

## Section 1: Study Overview

### 1.1 Purpose and Objectives

This study aims to construct RajNLP-50K, the first open, annotated corpus of Rajasthani-Hindi code-switched text for natural language processing (NLP) research. The corpus will contain 50,000 sentences collected from public social media platforms (Twitter/X and ShareChat) and annotated by native Rajasthani-Hindi bilingual speakers across three tasks:

1. **Sentiment analysis** — labeling sentences as positive, neutral, or negative
2. **Named entity recognition** — identifying person, location, and organization names
3. **Toxicity detection** — identifying caste-based slurs, religious hatred, gender-based harassment, and general toxic content

The annotated corpus will be released publicly on HuggingFace to support NLP research on low-resource Indian languages.

### 1.2 Scientific Justification

Rajasthani is spoken by approximately 50 million people but has virtually no NLP resources. Code-switching between Rajasthani and Hindi is the dominant mode of communication on social media in Rajasthan. Without annotated data, it is impossible to build NLP tools that serve this population. This corpus will enable:

- Sentiment analysis for public opinion monitoring
- Named entity recognition for information extraction
- Toxicity detection for content moderation, particularly for caste-based hate speech which is severely under-studied

---

## Section 2: Participant Information

### 2.1 Participant Population

- **Number of participants**: 3–5 annotators
- **Eligibility criteria**: Native Rajasthani-Hindi bilingual speakers, age 18+, able to read Devanagari script
- **Exclusion criteria**: Non-native speakers, minors, individuals with known trauma related to caste-based discrimination (self-reported)

### 2.2 Recruitment

Annotators will be recruited through:
- University linguistics department notice boards
- LinkedIn job postings
- Rajasthani cultural associations
- Referrals from existing participants

### 2.3 Compensation

Annotators will be compensated at:
- ₹150/hour for sentiment and NER tasks
- ₹200/hour for toxicity tasks (higher rate reflects the distressing nature of the content)

Payment will be made weekly via bank transfer.

---

## Section 3: Study Procedures

### 3.1 Annotation Process

1. Annotators will be onboarded via a written orientation document (Annotation Guidelines)
2. Before accessing toxicity content, annotators will read and acknowledge a written content warning
3. Annotation will be conducted through a locally hosted Label Studio instance
4. Each sentence will be labeled by all 3 annotators independently (blind annotation)
5. Inter-annotator agreement will be computed after each batch of 500 sentences
6. Batches below agreement thresholds will be adjudicated by an expert annotator

### 3.2 Data Handled by Participants

Annotators will encounter:
- General social media text (sentiment and NER tasks) — low risk
- Toxic content including caste-based slurs, religious hatred, gender-based harassment, and general abusive language (toxicity task) — **moderate to high risk of psychological distress**

### 3.3 Duration

- Total annotation time per participant: approximately 100–150 hours over 6–8 weeks
- Maximum toxicity annotation: 2 hours per day per annotator (enforced by the annotation system)

---

## Section 4: Risk Assessment

### 4.1 Risks

**Primary risk**: Psychological distress from sustained exposure to toxic content, particularly caste-based slurs which may be personally relevant to annotators from marginalized communities.

**Risk level**: Moderate

### 4.2 Risk Mitigation

1. **Content warning**: Written warning presented before any toxicity annotation begins
2. **Opt-out mechanism**: Annotators may withdraw from the toxicity task at any time without penalty; their sentences will be reassigned
3. **Daily exposure limit**: Maximum 2 hours of toxicity annotation per day, enforced by the annotation system
4. **Weekly check-ins**: The PI will contact each toxicity annotator at least once per week
5. **Support resources**: Annotators will be provided with mental health support contacts (iCall: 9152987821; Vandrevala Foundation: 1860-2662-345)
6. **Compensation premium**: Higher hourly rate for toxicity tasks acknowledges the additional burden

### 4.3 Benefits

- Direct compensation for annotators
- Contribution to NLP research that benefits the Rajasthani-speaking community
- Potential for co-authorship on resulting publications (for significant contributors)

---

## Section 5: Data Privacy and Confidentiality

### 5.1 Data Collected

- Annotation labels (sentiment, NER spans, toxicity categories) — linked to annotator ID
- Annotation timestamps
- Opt-out events (if any)

### 5.2 Data Storage

- All data stored on a password-protected local server
- Annotator identities stored separately from annotation data
- Data will not be shared with third parties

### 5.3 Publication

- Annotator identities will not be disclosed in any publication
- Aggregate demographics (age range, gender distribution, region) may be reported
- Annotators may be acknowledged by name with their explicit consent

### 5.4 Data Retention

- Annotation data will be retained indefinitely as part of the public corpus
- Personally identifiable information will be deleted after the study concludes

---

## Section 6: Informed Consent

All participants will provide written informed consent before beginning the study. The consent form will describe:
- The purpose of the study
- The nature of the content they will encounter
- Their right to withdraw at any time without penalty
- How their data will be used and stored
- Compensation details

A separate consent form will be used for the toxicity annotation task, given the higher risk level.

---

## Section 7: Attachments

- [ ] Annotation Guidelines document
- [ ] Content warning text
- [ ] Informed consent form (general)
- [ ] Informed consent form (toxicity task)
- [ ] Annotator screening questionnaire
- [ ] Sample annotation interface screenshots

---

*Prepared by: [Your Name]*  
*Date: [Date]*  
*Institution: [Institution Name]*
