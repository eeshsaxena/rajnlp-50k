# IRB Application Template — RajNLP-50K

**Fill in all [BRACKETED] fields before submitting to your institution's IRB.**

---

## Section 1: Project Information

**Project Title:** RajNLP-50K: Building an Annotated Rajasthani-Hindi Code-Switched NLP Corpus

**Principal Investigator:** [YOUR FULL NAME]  
**Institution:** [YOUR INSTITUTION NAME]  
**Department:** [YOUR DEPARTMENT]  
**Email:** [YOUR EMAIL]  
**Phone:** [YOUR PHONE]  
**Supervisor/Advisor:** [ADVISOR NAME, if applicable]

**Proposed Start Date:** [DATE]  
**Proposed End Date:** [DATE]  
**Funding Source:** [SELF-FUNDED / GRANT NAME / INSTITUTION]

---

## Section 2: Project Summary

This project constructs the first open, annotated Rajasthani-Hindi code-switched NLP corpus (RajNLP-50K) containing 50,000 sentences sourced from public social media platforms (Twitter/X and ShareChat). The corpus will be annotated for sentiment analysis, named entity recognition, and toxicity detection by 3 native Rajasthani-Hindi bilingual speakers.

The research aims to:
1. Create a publicly available linguistic resource for the under-resourced Rajasthani language
2. Train and evaluate NLP models for code-switched text
3. Develop the first caste-based toxicity classifier for Rajasthani-Hindi text
4. Publish findings at a peer-reviewed NLP venue (ACM TALLIP, EMNLP, or LREC-COLING)

---

## Section 3: Human Subjects Involvement

### 3.1 Who are the participants?

**Annotators (n=3):**
- Native Rajasthani-Hindi bilingual speakers
- Adults aged 18 or older
- Recruited through [university notice boards / LinkedIn / personal networks]
- Compensated at ₹150/hour (sentiment, NER) and ₹200/hour (toxicity)

**Social media users (indirect):**
- Publicly posted content from Twitter/X and ShareChat will be collected
- No direct interaction with social media users
- Only public posts will be used; no private messages or accounts

### 3.2 Recruitment procedure

Annotators will be recruited via:
- [University linguistics/Hindi department notice boards]
- [LinkedIn job posting]
- [Personal/professional networks]

A screening test will verify Rajasthani-Hindi bilingual proficiency before onboarding.

### 3.3 Consent procedure

All annotators will:
1. Receive a written information sheet describing the project
2. Receive a written content warning describing toxic content they will encounter
3. Sign a written informed consent form before beginning annotation
4. Be informed of their right to withdraw at any time without penalty

---

## Section 4: Risks and Benefits

### 4.1 Risks

**Annotators:**
- **Psychological distress** from exposure to toxic content (caste slurs, religious hatred, gender-based harassment). Mitigation: (a) written content warning before onboarding, (b) opt-out mechanism at any time, (c) daily 2-hour limit on toxicity annotation, (d) weekly wellbeing check-ins with the annotation lead.
- **Privacy:** Annotators' identities will not be published. Only aggregate demographics will be reported.

**Social media users:**
- **Privacy:** Only publicly posted content will be collected. No personally identifiable information (names, account handles, profile photos) will be published in the corpus. All text will be anonymized before release.
- **Minimal risk:** Collection is limited to public posts; no interaction with users.

### 4.2 Benefits

- Annotators receive fair compensation and gain research experience
- The NLP community gains a publicly available resource for an under-resourced language
- Potential downstream benefit for Rajasthani language preservation and digital inclusion

---

## Section 5: Data Collection

### 5.1 What data will be collected?

**From social media:**
- Text content of public posts (no images, videos, or audio)
- Post URL (for provenance)
- Collection timestamp
- Platform identifier (Twitter/X or ShareChat)

**No collection of:**
- User names, handles, or profile information
- Private messages
- Location data
- Images or multimedia

### 5.2 Data storage and security

- Raw data stored on [encrypted local storage / university server]
- Access restricted to the PI and annotation team
- Data will be anonymized (URLs removed or hashed) before public release
- Annotator identities stored separately from annotation data

### 5.3 Data retention

- Raw (unanonymized) data: deleted after corpus construction is complete
- Anonymized corpus: retained indefinitely and published publicly on HuggingFace
- Annotator consent forms: retained for [5 years] per institutional policy

---

## Section 6: Toxicity Content Handling

Given that this project involves annotation of toxic content including caste slurs, religious hatred, and gender-based harassment, the following additional protections are in place:

1. **Written content warning** presented to all annotators before they access the toxicity annotation task
2. **Opt-out mechanism** allowing annotators to withdraw from the toxicity task at any time without penalty, with their sentences reassigned to a replacement annotator
3. **Daily exposure limit** of 2 hours of toxicity annotation per annotator per day
4. **Weekly wellbeing check-ins** conducted by the annotation lead
5. **Higher compensation** (₹200/hour vs ₹150/hour) for toxicity annotation to reflect the additional burden
6. **Annotator support resources** provided (counseling contacts, mental health resources)

---

## Section 7: Anonymization Plan

Before public release, the corpus will be anonymized as follows:

1. **Source URLs** will be removed or replaced with a hash
2. **Named entities** (person names, account handles) in the text will NOT be removed (they are part of the linguistic data), but will be clearly labeled as named entities in the annotation
3. **Annotator identities** will not be published; only aggregate demographics (age range, gender distribution, region) will be reported in the paper
4. The dataset card will include a statement that the corpus contains real social media text that may include names of public figures

---

## Section 8: Checklist

- [ ] Informed consent form attached
- [ ] Annotator information sheet attached
- [ ] Content warning text attached (see `annotator_tool/welfare.py`)
- [ ] Annotation guidelines attached (see `docs/annotation_guidelines.md`)
- [ ] Data storage plan described
- [ ] Anonymization plan described
- [ ] Compensation rates specified
- [ ] Opt-out procedure described

---

## Attachments

1. Informed Consent Form (to be drafted)
2. Annotator Information Sheet (to be drafted)
3. Content Warning Text (see `annotator_tool/welfare.py` → `CONTENT_WARNING`)
4. Annotation Guidelines (`docs/annotation_guidelines.md`)
5. Screening Test for Annotators (`docs/annotator_screening_test.md`)
