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
- 
![lab3 2](https://user-images.githubusercontent.com/75094394/196775234-3cef04ca-8660-4590-ac8c-79504567afa1.gif)


