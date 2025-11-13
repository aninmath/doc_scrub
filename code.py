
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables if needed
load_dotenv()

# Step 1: Load PDF text
loader = PyMuPDFLoader("swm.pdf")
docs = loader.load()
text = "\n".join([doc.page_content for doc in docs])

# Step 2: Define Pydantic models



class Marine(BaseModel):
    sched_datetime: str = Field(description="Scheduled date and time")
    status: Optional[str] = Field(default=None, description="Status of the schedule (e.g., SCH)")
    vessel: Optional[str] = Field(default=None, description="Name of the vessel")
    vss_num: Optional[str] = Field(default=None, description="VSS number")
    berth: Optional[str] = Field(default=None, description="Berth identifier")
    direction: Optional[str] = Field(default=None, description="Inbound or Outbound")
    trans_id: Optional[str] = Field(default=None, description="Transaction ID")
    proj_total_vol: Optional[str] = Field(default=None, description="Projected total volume")
    inspector: Optional[str] = Field(default=None, description="Inspector name")
    agent: Optional[str] = Field(default=None, description="Agent name")
    type: Optional[str] = Field(default=None, description="Type")
    io: Optional[str] = Field(default=None, description="IO field")
    comments: Optional[str] = Field(default=None, description="Comments")
    products: List[str] = Field(default=[], description="List of products involved")
    customers: List[str] = Field(default=[], description="List of customers involved")
    notes: List[str] = Field(description="Additional notes")




class MarineList(BaseModel):
    items: List[Marine] = Field(description="List of Marine schedule entries")

# Step 3: Create parser for MarineList
parser = PydanticOutputParser(pydantic_object=MarineList)

# Step 4: Create prompt template with format instructions
prompt = PromptTemplate(
    input_variables=["text"],
    template="""

Extract all schedule entries from the text and return as JSON matching this schema:
{format_instruction}

Important:
- 'vessel' is the value after 'Status' and before 'VSS Num'.
- 'trans_id' is the value after 'Direction' and before 'Proj Total Vol'.
- 'vss_num' is the value after 'Vessel' and before 'Berth'.

Text:
{text}

""",
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Step 5: Initialize Gemini model
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key= 'AIzaSyCS36bFxt4Y08dqom9MqrsGNEStZ6MmIPM')

# Step 6: Create chain
chain = prompt | model | parser

# Step 7: Invoke chain
result = chain.invoke({'text': text})

# Step 8: Convert to DataFrame
df = pd.DataFrame([entry.dict() for entry in result.items])
print(f"✅ Extracted {len(df)} rows")
print(df.head())

# Optional: Save to CSV
df.to_csv("marine_schedule_ai.csv", index=False)