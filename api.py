from requests import get
lapa_length = input('Введите длину лапы = ')
lapa_width = input('Введите ширину лапы = ')
print(get(f'http://localhost:5000/api?lapa_length={lapa_length}&lapa_width={lapa_width}').json())