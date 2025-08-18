from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from .database import init_db, get_session
from .models import Field, InputItem, Operation

app = FastAPI()

@app.on_event("startup")
def on_startup() -> None:
    init_db()

@app.post("/fields/", response_model=Field)
def create_field(field: Field, session: Session = Depends(get_session)) -> Field:
    session.add(field)
    session.commit()
    session.refresh(field)
    return field

@app.get("/fields/", response_model=List[Field])
def read_fields(session: Session = Depends(get_session)) -> List[Field]:
    return session.exec(select(Field)).all()

@app.post("/inputs/", response_model=InputItem)
def create_input(item: InputItem, session: Session = Depends(get_session)) -> InputItem:
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

@app.post("/operations/", response_model=Operation)
def create_operation(op: Operation, session: Session = Depends(get_session)) -> Operation:
    if op.input_id:
        item = session.get(InputItem, op.input_id)
        if item:
            item.stock -= op.quantity
    session.add(op)
    session.commit()
    session.refresh(op)
    return op

@app.get("/fields/{field_id}/cost")
def compute_cost(field_id: int, session: Session = Depends(get_session)) -> dict:
    ops = session.exec(select(Operation).where(Operation.field_id == field_id)).all()
    total = 0.0
    for op in ops:
        if op.input_id:
            item = session.get(InputItem, op.input_id)
            if item:
                total += op.quantity * item.unit_price
        total += op.fuel
    return {"field_id": field_id, "total_cost": total}

# Serve static files including index.html
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent, html=True), name="static")
