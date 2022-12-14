# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Голованов Богдан Михайлович 
- ХИИ31

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Задание 2.
- Задание 3.
- Выводы.

##Цель работы: 
Познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity.

- Создаю новый пустой 3D проект на Unity.
- В созданный проект добавляю ML Agent.
- Далее запускаю Anaconda Prompt для установки пакетов mlagents 0.28.0 и torch 1.7.1
- Создаю на сцене плоскость, куб и сферу по примеру.

![Unity_FCWYeIIFxV](https://user-images.githubusercontent.com/75094394/196740496-474a0b81-4078-49bc-b5c0-5240a5f50358.png)

- Создаю C# скрипт и подключаю его к сфере.
- В скрипт-файл RollerAgent.cs добавляю код, опубликованный в материалах лабораторных работ.

```py

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

```

- Объекту «сфера» добавил компоненты Decision Requester, Behavior Parameters

![Unity_XfRxGu7vpZ](https://user-images.githubusercontent.com/75094394/196744263-04f7bd1d-5e21-4746-8496-a4e2f93e1d7e.png)

- В корень проекта добавил файл конфигурации нейронной сети.

```yaml

behaviors:
  RollerBall:
    trainer_type: ppo
    hyperparameters:
      batch_size: 10
      buffer_size: 100
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000

```
- Запустил работу mlagents
![cmd_dz69xjrNdA](https://user-images.githubusercontent.com/75094394/196770206-0bc5ee9e-37a7-43a3-a03e-42867bda3574.png)

- 1 копия 

![lab3](https://user-images.githubusercontent.com/75094394/196770675-69ed99c7-5b12-4d41-bf69-ab5ed9bfd5d1.gif)

- 27 копий

![lab3 1](https://user-images.githubusercontent.com/75094394/196773234-9100cbfc-f495-4ae6-b46d-4890fd8fd5bb.gif)

![cmd_r8QC003R91](https://user-images.githubusercontent.com/75094394/196774026-5ca26fdf-3cb5-45b2-bf52-697f49cd5c73.png)


- После завершения обучения проверил работу модели 

![lab3 2](https://user-images.githubusercontent.com/75094394/196775234-3cef04ca-8660-4590-ac8c-79504567afa1.gif)

- Вывод. Модель натренировалась ссредней наградой - 0.994, Кубики собирает хорошо, но все же может выпасть за поле, если кубик находится на краю.

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

- Файл конфигурации rollerball_config.yam

```yaml

behaviors: #Создание списка "Модель поведения" дял разных агентов.
  RollerBall:  #Создание списка конкретного объекта.
    trainer_type: ppo  #Тип используемого тренажера, ppo - это алгоритм обучения с подкреплением от OpenAI.
    hyperparameters:
      batch_size: 10 #Количество опытов на каждой итерации градиентного спуска.
      buffer_size: 100 #Количество опыта, которое необходимо собрать перед обновлением модели политики.
      learning_rate: 3.0e-4    # Начальная скорость обучения для градиентного спуска. С
      beta: 5.0e-4 #Сила регуляризации энтропии, которая делает политику "более случайной".
      epsilon: 0.2 #Влияет на то, насколько быстро политика может развиваться во время обучения.
      lambd: 0.99 #Насколько агент полагается на свою текущую оценку значений при расчете предсказаний. Высокие значения соответствуют тому, что агент больше полагается на фактические вознаграждения, полученные в окружающей среде
      num_epoch: 3  #Количество проходов, которые необходимо выполнить через буфер опыта при выполнении оптимизации градиентного спуска.
      learning_rate_schedule: linear  #Определяет, как скорость обучения меняется с течением времени.
    network_settings:    #Настройки нейронной сети.
      normalize: false   #Применяется ли нормализация к входным данным векторного наблюдения. 
      hidden_units: 128     #Количество нейронов в скрытых слоях нейронной сети.
      num_layers: 2      #Количество скрытых слоев в нейронной сети. Соответствует количеству скрытых слоев, присутствующих после входящих данных
    reward_signals:    #Настройки для внешних и внутренних сигналов вознаграждения
      extrinsic: #Внешние награды.
        gamma: 0.99   #Коэффициент дисконтирования для будущих вознаграждений, поступающих от окружающей среды. Это можно рассматривать как то, насколько далеко в будущем агент должен заботиться о возможных вознаграждениях.
        strength: 1.0   #Значение, на которое можно умножить вознаграждение, получаемое от окружающей среды.
    max_steps: 500000    #Общее количество шагов до завершения обучения.
    time_horizon: 64     #Сколько шагов опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер опыта.
    summary_freq: 10000    #Количество опыта, которое необходимо собрать перед созданием и отображением статистики обучения.

``` 
- Параметр Decision Period определяет частоту, с которой агент запрашивает решение.
- Период принятия решения, равный n, означает, что Агент будет запрашивать решение каждые n шагов обучения.

Компонент Behavior Parameters (параметры поведения) определяет, как объект принимает решения.

- Behavior Name. Имя этого поведения, которое используется в качестве базового имени и указывается в файле конфигурации модели.
- Behavior Type. Определяет, какой тип поведения будет использовать Агент. Default - Агент будет использовать удаленный процесс обучения, запущенный через python для принятия решений. InferenceOnly агент всегда будет использовать предоставленную моделью нейронной сети. HeuristicOnly - всегда используется эвристический метод.
- Model. Используемая модель нейронной сети.
- InferenceDevice. Выбор между CPU и GPU для предоставленной модели нейронной сети.
- Vector Observation. Вектор наблюдения - это вектор чисел с плавающей запятой, которые содержат релевантную информацию для принятия агентом решений. Вектор заполняется в функции CollectObservations.

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости.

- Доработал сцену

![Unity_35s9FcJFuU](https://user-images.githubusercontent.com/75094394/196978957-59089667-472a-43d4-b7ce-e6d5aab25246.png)


В класс RollerAgent добавил вторую цель, а также парамет, который говорит о том, что был собран кубик.
В функции OnEpisodeBegin теперь позиция задаётся двум кубам случайным образом.
В функцию CollectObservation добавил считывание параметра, отвечающего за то, был собран каждый куб.
Функция OnActionReceived на основе дистации до каждого из кубиков определяет, было ли попадание. В случае попадания кубик скрывается. 
Если два кубика были сбиты или шарик выпал, то эпизод заканчивается.

    public override void OnActionReceived(ActionBuffers actionBuffers)
 ```py

  public GameObject Target1;
    private bool active1 = false;
    public GameObject Target2;
    private bool active2 = false;
    private float revard = 0.0f;

    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target1.transform.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
        Target2.transform.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);

        revard = 0.0f;
        active1 = true;
        active2 = true;
        Target1.SetActive(true);
        Target2.SetActive(true);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target1.transform.localPosition);
        sensor.AddObservation(Target2.transform.localPosition);
        sensor.AddObservation(active1);
        sensor.AddObservation(active2);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }

    public float forceMultiplier = 10;

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget1 = Vector3.Distance(this.transform.localPosition, Target1.transform.localPosition);
        float distanceToTarget2 = Vector3.Distance(this.transform.localPosition, Target2.transform.localPosition);


        if (active1 && distanceToTarget1 < 1.42f)
        {
            active1 = false;
            revard += 0.5f;
            Target1.SetActive(false);
        }

        if (active2 && distanceToTarget2 < 1.42f)
        {
            active2 = false;
            revard += 0.5f;
            Target2.SetActive(false);
        }

        if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
        else if ((!active1 && !active2))
        {
            SetReward(revard);
            EndEpisode();
        }
    }

 ```
 
![cmd_DCSncOKJcR](https://user-images.githubusercontent.com/75094394/196980796-824b34a8-0263-47be-80bd-880b1d44c2e6.png)
 
 - Результат обучненной модели

![lab3 3](https://user-images.githubusercontent.com/75094394/196980669-2683cbd5-a147-413f-96b1-81d48284de6f.gif)

## Вывод
### В выводах к работе дайте развернутый ответ, что такое игровой баланс и как системы машинного обучения могут быть использованы для того, чтобы его скорректировать.

Игровой баланс — «равновесие» между персонажами, командами, тактиками игры и другими игровыми объектами. Игровой баланс — одно из требований к «честности» правил. Хороший баланс должен поддерживать уровень "фана" в небольшом диапазоне, когда игра не кажется слишком простой или не настолько сложная, что хочется её выключить. При этом игрок должен ощущать, что игра честная по отношению к нему. Обычно баланс игры в первых версиях задают на интуитивном уровне. Есть разнообразные приёмы балансировки. Например, мощность персонажа можно расчитать, как сумму или среднее его характеристик и сталкивать персонажей с похожей мощностью. Можно замерять HPS и DPS и за мощность считать их произведение. Или можно прибегнуть к визуализацию характеристик на графике и интуитивно их корректировать.

Машинное обучение используется для настройки игрового баланса. Проводя при помощи обученных агентов миллионы симуляций с целью сбора данных, эта методика тестирования игр на основе ML позволяет эффективнее повышать интересность и баланс игр. Каждый игровой персонаж, которым ранее должен был управлять человек, заменяется на агента, которому доступны те же способности и возможности, что и реальному игроку. Сталкивая таких объектов тысячи раз можно выявить какие-то ультрасильные стратегии поведения и бороться с ними.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
