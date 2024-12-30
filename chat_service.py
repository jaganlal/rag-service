from fastapi import status, APIRouter
from pydantic_models import QueryInput, QueryResponse
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history
import uuid
import logging

router = APIRouter(
    prefix='/api/1.0/rag-chat-service',
    tags=['Start the chat service'],
)

@router.post(
  '/chat',
  summary = "Begin the chat service",
  description = "Begin the chat service",
  response_model = QueryResponse,
  status_code=status.HTTP_200_OK
  )
def chat(query_input: QueryInput):
  session_id = query_input.session_id or str(uuid.uuid4())
  logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

  chat_history = get_chat_history(session_id)
  rag_chain = get_rag_chain(query_input.model.value)
  answer = rag_chain.invoke({
      "input": query_input.question,
      "chat_history": chat_history
  })['answer']

  insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
  logging.info(f"Session ID: {session_id}, AI Response: {answer}")
  return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)
