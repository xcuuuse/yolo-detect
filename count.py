import json

# 1. Открываем и читаем JSON-файл
with open('data/result_yolo26x_cuda.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


total_people = sum(value for key, value in data.items() if key != 'metrics')

print(f"Сумма всех найденных объектов за тест: {total_people}")