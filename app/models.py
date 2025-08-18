from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class Field(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    geometry: str  # GeoJSON string
    crop: Optional[str] = None
    operations: List["Operation"] = Relationship(back_populates="field")

class InputItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    unit: str
    stock: float = 0
    unit_price: float = 0

class Operation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    field_id: int = Field(foreign_key="field.id")
    input_id: Optional[int] = Field(default=None, foreign_key="inputitem.id")
    quantity: float = 0
    fuel: float = 0

    field: Field = Relationship(back_populates="operations")
    input_item: Optional[InputItem] = Relationship()
