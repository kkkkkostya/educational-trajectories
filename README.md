# Программа прогнозирования образовательных траекторий на основе трансформерных нейронных сетей
**Данный программа представляет из себя веб-приложение, с помощью которого пользователь может загрузить ведомость с оценками в разных форматах (на данный момент поддерживаются csv, excel и txt форматы) и получить результат работы модели и скачать его себе на устройство**

- ```Для работы веб-приложения необходимо локально запустить его.```
  
Оно состоит из двух небольших сервисов:
>  Непосредственно [веб-приложение](https://github.com/kkkkkostya/educational-trajectories/tree/e3b65c17cdbb33ab13dd94c2b3cf0cb754983b07/production/streamlit-service) (command to run: "streamlit run streamlit_app.py" или можно также запустить отдельно [docker контейнер](https://github.com/kkkkkostya/educational-trajectories/blob/7fd06e55c8f0cec8e8d54818eaeb3faa9577c47f/production/streamlit-service/Dockerfile)) <br/>
>  [FastApi сервис](https://github.com/kkkkkostya/educational-trajectories/tree/main/production/models/model-api) для работы моделей (command to run: python -m model_api.model.api или можно также запустить отдельно [docker контейнер](https://github.com/kkkkkostya/educational-trajectories/blob/7fd06e55c8f0cec8e8d54818eaeb3faa9577c47f/production/models/Dockerfile))
- ```После запуска приложения пользователю доступны разделы:```<br/>
  - ```Welcone info (общая информация о проекте и разделах придложения)```
  - ```Scores prediction (основная страница для предсказания пропущенных значений в данных)```
  - ```Prediction result (визуализация работы модели и возможно локально загрузить результат в разных форматах)```
