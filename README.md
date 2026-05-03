# DocAI

Extract structured data from  unstructured documents using Answer.AI's [Byaldi](https://github.com/AnswerDotAI/byaldi), OpenAI [gpt-4o](https://platform.openai.com/docs/guides/vision), and [Langchain's structured output](https://python.langchain.com/v0.1/docs/modules/model_io/chat/structured_output/).

### Installation
```bash
pyenv virtualenv 3.10.6 docai
pyenv activate docai
poetry install
```

### Environment vars
Ensure you have an OPENAI_API_KEY and HF_TOKEN set in your environment variables.

```bash
export OPENAI_API_KEY=<your key>
export HF_TOKEN=<your token>
```

### Sample usage

Build the index from the pdfs/ folder:
```bash
python scripts/build_index.py --folder "pdfs/" --index_name "application"
```

Extraction structured information from the index (open extract.py to see queries and pydantic models):
```bash
python scripts/extract.py
```

To add OCR text context from Reducto Parse for the pages retrieved from the index:
```bash
export REDUCTO_API_KEY=<your key>
python scripts/extract.py --ocr_provider reducto
```

You can also enable it in code:
```python
extractor = Extractor(index_name="application", ocr_provider="reducto")
```

### Sample output
```
What losses have occurred in the past 5 years?
LossHistory(
    losses=[
        Loss(loss_date='2/20/21', loss_amount=7003.0, loss_description='Claimant was in his sleeper when his truck got hit by insured driver on the left', date_of_claim='4/19/21'),
        Loss(loss_date='2/4/21', loss_amount=92584.0, loss_description='The IV was attempting to merge on the highway when the IV lost control and struck', date_of_claim='4/30/21'),
        Loss(loss_date='9/14/21', loss_amount=5583.0, loss_description='IV was in the fast lane, when IV tire flew off and struck OV1, OV2, OV3, OV4', date_of_claim='9/15/21'),
        Loss(loss_date='9/14/21', loss_amount=6299.0, loss_description='IV was in the fast lane, when IV tire flew off and struck OV1, OV2, OV3, OV4', date_of_claim='9/15/21')
    ]
)

What is the basic application information?
Application(
  insured_name='Greentown Burgers LLC', 
  insured_address='Not provided', 
  insured_phone='Not provided',
  insured_email='Not provided', 
  effective_date='07/22/2024'
)
```
