from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None]=None

@app.get('/')
def read_root():
    return {'Hello': 'world'}


@app.get('/items/{item_id}')
def read_item(item_id: int, q: str):
    return {'item_id': item_id, 'q': q}


@app.put('/items/{item_id}')
def update_item(item_id: int, item: Item):
    return {'item)id': item_id, 'item': item.price}
