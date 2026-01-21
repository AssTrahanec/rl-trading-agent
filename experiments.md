# Experiment Log

Документация всех экспериментов с RL Trading Agent.

---

## Experiment Run #1 (Baseline)

**Дата:** 2025-01-21
**Цель:** Установить baseline и сравнить разные подходы к reward и нормализации
**Данные:** Синтетические (500 дней, seed=42)
**Обучение:** PPO, 50K timesteps (5 epochs × 10K)

### Конфигурации

| Experiment | Reward Type | Normalize Obs | Entropy | Transaction Penalty |
|------------|-------------|---------------|---------|---------------------|
| baseline | simple_pnl | False | 0.01 | 0.0 |
| improved_reward | sharpe_based | False | 0.01 | 0.001 |
| normalized_obs | simple_pnl | True | 0.01 | 0.0 |
| combo | sharpe_based | True | 0.02 | 0.001 |

### Результаты

| Experiment | Return | vs Buy&Hold | Sharpe | Drawdown | Win Rate | Trades |
|------------|--------|-------------|--------|----------|----------|--------|
| **normalized_obs** | +123.65% | **+52.22%** | 2.35 | **-8.66%** | 30.51% | 152 |
| baseline | +108.13% | +36.70% | 1.87 | -11.24% | 40.53% | 298 |
| combo | +101.17% | +29.74% | **2.40** | -12.75% | **66.15%** | 139 |
| improved_reward | +73.05% | +1.62% | 1.37 | -12.79% | 51.67% | 309 |

**Buy & Hold baseline:** +71.43%

### Анализ

#### Что работает:
1. **Нормализация observations** - критически важна! Без неё sharpe_based reward почти не работает
2. **Simple PnL + Normalized** - лучший return (+123.65%) и лучший drawdown (-8.66%)
3. **Combo** - лучший Sharpe (2.40) и win rate (66.15%), хороший баланс

#### Что не работает:
1. **Sharpe-based без нормализации** - outperformance всего +1.62%, хуже baseline
2. Transaction penalty без нормализации мешает обучению

#### Action Distribution паттерны:
- **normalized_obs**: Агрессивная стратегия - 46% SELL_100%, 33% BUY_50%, почти не использует HOLD
- **combo**: Сбалансированная - 17.8% HOLD, использует разные уровни buy/sell
- **baseline**: Много мелких сделок (298), активно использует SELL_25%

### Выводы для следующих экспериментов:

1. Всегда использовать `normalize_obs=True`
2. Попробовать увеличить timesteps (100K+) для лучшей сходимости
3. Попробовать разные learning rates
4. Исследовать влияние entropy coefficient на exploration
5. Попробовать более агрессивные transaction penalties с нормализацией

---

## Experiment Run #2 (Optimization)

**Дата:** 2025-01-21
**Цель:** Оптимизация на основе выводов Run #1
**Гипотезы:**
1. Больше timesteps → лучшая сходимость
2. Меньший learning rate → более стабильное обучение
3. Выше entropy → больше exploration → возможно лучшие стратегии
4. Transaction penalty с нормализацией может улучшить Sharpe

### Новые конфигурации

| Experiment | Базируется на | Изменения |
|------------|---------------|-----------|
| norm_100k | normalized_obs | timesteps: 50K → 100K |
| norm_low_lr | normalized_obs | learning_rate: 3e-4 → 1e-4 |
| norm_high_ent | normalized_obs | entropy_coef: 0.01 → 0.05 |
| norm_with_penalty | normalized_obs | +transaction_penalty: 0.002 |

### Результаты

| Experiment | Return | vs Buy&Hold | Sharpe | Drawdown | Win Rate | Trades |
|------------|--------|-------------|--------|----------|----------|--------|
| **norm_100k** | **+187.20%** | **+115.76%** | 2.59 | -9.68% | 35.56% | 222 |
| norm_high_ent | +155.84% | +84.41% | **2.75** | **-7.40%** | **41.87%** | 179 |
| normalized_obs (baseline) | +118.92% | +47.49% | 2.31 | -8.11% | 35.06% | 229 |
| norm_with_penalty | +109.49% | +38.06% | 2.38 | -8.33% | 36.36% | 187 |
| norm_100k_low_lr | +102.23% | +30.80% | 1.89 | -9.88% | 38.54% | 223 |
| norm_low_lr | +89.19% | +17.76% | 1.78 | -11.65% | 35.29% | 229 |

**Buy & Hold baseline:** +71.43%

### Анализ

#### Ключевые находки:

1. **Больше timesteps = значительно лучше** ✅
   - norm_100k (+187%) vs baseline (+118%) = +69% improvement
   - Гипотеза подтверждена

2. **Высокий entropy улучшает качество, но не return** ✅
   - norm_high_ent: лучший Sharpe (2.75) и лучший Win Rate (41.87%)
   - Но return ниже чем у norm_100k
   - Больше exploration → более стабильная стратегия

3. **Низкий learning rate = ХУЖЕ** ❌
   - norm_low_lr (+89%) vs baseline (+118%) = -29% regression
   - norm_100k_low_lr (+102%) vs norm_100k (+187%) = -85% regression
   - Гипотеза отвергнута! Меньший LR не помогает

4. **Transaction penalty незначительно влияет**
   - norm_with_penalty (+109%) близко к baseline (+118%)
   - Не улучшает и не сильно портит

#### Что работает:
- ✅ Больше timesteps (100K >> 50K)
- ✅ Высокий entropy (0.05) для лучшего Sharpe/Win Rate
- ✅ Стандартный learning rate (3e-4)

#### Что не работает:
- ❌ Низкий learning rate (1e-4) - ухудшает результаты
- ⚠️ Transaction penalty - нейтральный эффект

---

## Experiment Run #3 (Scaling Up)

**Дата:** 2025-01-21
**Цель:** Комбинировать лучшие находки и масштабировать
**Гипотезы:**
1. 100K + high entropy = лучшее из обоих миров (return + Sharpe)
2. 200K timesteps = еще лучше чем 100K
3. Более высокий LR (5e-4) может ускорить сходимость

### Новые конфигурации

| Experiment | Базируется на | Изменения |
|------------|---------------|-----------|
| combo_best | norm_100k | +entropy_coef: 0.05 (100K + high entropy) |
| steps_200k | norm_100k | timesteps: 100K → 200K |
| high_lr | normalized_obs | learning_rate: 3e-4 → 5e-4 |
| steps_200k_ent | steps_200k | +entropy_coef: 0.05 (максимум) |

### Результаты

*Ожидают запуска...*

| Experiment | Return | vs Buy&Hold | Sharpe | Drawdown | Win Rate | Trades |
|------------|--------|-------------|--------|----------|----------|--------|
| combo_best | - | - | - | - | - | - |
| steps_200k | - | - | - | - | - | - |
| high_lr | - | - | - | - | - | - |
| steps_200k_ent | - | - | - | - | - | - |

### Анализ

*После получения результатов...*

---

## Summary Table (All Runs)

| Run | Best Experiment | Return | Sharpe | Drawdown | Key Finding |
|-----|-----------------|--------|--------|----------|-------------|
| #1 | normalized_obs | +123.65% | 2.35 | -8.66% | Нормализация критична |
| #2 | norm_100k | +187.20% | 2.59 | -9.68% | Больше timesteps = лучше |
| #3 | TBD | - | - | - | - |

---

## Hyperparameter Reference

### PPO Defaults (текущие)
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01  # varies by experiment
```

### Training Defaults
```python
total_timesteps = 50000  # 5 epochs × 10K
```

### Environment
```python
initial_balance = 10000
commission = 0.001  # 0.1%
observation_space:
  - market: 15 features
  - news: 6 features
  - portfolio: 5 features
action_space: Discrete(7)  # HOLD, BUY 25/50/100%, SELL 25/50/100%
```
