import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", message=".*protected namespace.*")

import aiohttp
import aiosqlite
import emoji
from aiohttp_socks import ProxyConnector
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BufferedInputFile, Message
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DB_PATH = os.getenv("DB_PATH", "bot_data.db")
PROXY_URL = os.getenv("PROXY_URL")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в переменных окружения!")

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()

KCAL_PER_MIN_PER_KG = 3.5 / 200

WORKOUTS = {
    "бег": {"met": 9.0, "emoji": "🏃"},
    "ходьба": {"met": 3.5, "emoji": "🚶"},
    "плавание": {"met": 6.0, "emoji": "🏊"},
    "велосипед": {"met": 8.0, "emoji": "🚴"},
    "йога": {"met": 2.5, "emoji": "🧘"},
    "силовая": {"met": 3.5, "emoji": "🏋️"},
    "кардио": {"met": 7.0, "emoji": "💪"},
    "танцы": {"met": 5.0, "emoji": "💃"},
    "теннис": {"met": 8.0, "emoji": "🎾"},
    "баскетбол": {"met": 6.5, "emoji": "🏀"},
    "футбол": {"met": 7.0, "emoji": "⚽"},
    "волейбол": {"met": 4.0, "emoji": "🏐"},
    "скакалка": {"met": 12.0, "emoji": "🩢"},
    "гребля": {"met": 7.0, "emoji": "🚣"},
    "лыжи": {"met": 8.0, "emoji": "⛷️"},
    "скалолазание": {"met": 6.5, "emoji": "🧗"},
}

WORKOUT_ALIASES = {
    "running": "бег",
    "run": "бег",
    "walk": "ходьба",
    "swim": "плавание",
    "bike": "велосипед",
    "yoga": "йога",
    "gym": "силовая",
    "cardio": "кардио",
    "dance": "танцы",
}


async def translate_to_english(text: str) -> str:
    if text.isascii():
        return text
    url = "https://api.mymemory.translated.net/get"
    params = {"q": text, "langpair": "ru|en"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    translated = data.get("responseData", {}).get("translatedText", "")
                    if translated and translated.lower() != text.lower():
                        return translated
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
    return text


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                weight REAL,
                height REAL,
                age INTEGER,
                gender TEXT,
                activity INTEGER DEFAULT 0,
                city TEXT,
                water_goal INTEGER DEFAULT 2000,
                calorie_goal INTEGER DEFAULT 2000,
                logged_water INTEGER DEFAULT 0,
                logged_calories REAL DEFAULT 0,
                burned_calories REAL DEFAULT 0,
                last_reset TEXT
            )
        """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT,
                water_logged INTEGER DEFAULT 0,
                water_goal INTEGER DEFAULT 2000,
                calories_logged REAL DEFAULT 0,
                calories_burned REAL DEFAULT 0,
                calorie_goal INTEGER DEFAULT 2000,
                UNIQUE(user_id, date)
            )
        """
        )
        await db.commit()


async def get_user_data(user_id: int) -> dict:
    today = datetime.now().date().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                await db.execute(
                    """INSERT INTO users (user_id, last_reset) VALUES (?, ?)""",
                    (user_id, today),
                )
                await db.commit()
                return {
                    "weight": None,
                    "height": None,
                    "age": None,
                    "gender": None,
                    "activity": 0,
                    "city": None,
                    "water_goal": 2000,
                    "calorie_goal": 2000,
                    "logged_water": 0,
                    "logged_calories": 0,
                    "burned_calories": 0,
                    "last_reset": today,
                }
            user_data = dict(row)
            if user_data.get("last_reset") != today:
                await db.execute(
                    """UPDATE users SET logged_water = 0, logged_calories = 0,
                       burned_calories = 0, last_reset = ? WHERE user_id = ?""",
                    (today, user_id),
                )
                await db.commit()
                user_data["logged_water"] = 0
                user_data["logged_calories"] = 0
                user_data["burned_calories"] = 0
                user_data["last_reset"] = today
            return user_data


async def update_user_data(user_id: int, **kwargs):
    if not kwargs:
        return
    set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [user_id]
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE users SET {set_clause} WHERE user_id = ?", values)
        await db.commit()


def calories_per_min(activity: str, weight_kg: float = 70) -> float:
    met = WORKOUTS[activity]["met"]
    return met * weight_kg * KCAL_PER_MIN_PER_KG


def get_workout_data(workout_type: str) -> tuple[str, dict]:
    workout_type = workout_type.lower().strip()
    if workout_type in WORKOUT_ALIASES:
        workout_type = WORKOUT_ALIASES[workout_type]
    if workout_type in WORKOUTS:
        return workout_type, WORKOUTS[workout_type]
    return workout_type, {"met": 4.0, "emoji": "🏋️"}


async def get_food_emoji(name: str) -> str:
    translated = await translate_to_english(name.lower())
    words = translated.replace(",", " ").replace("-", " ").split()
    for word in words:
        shortcode = f":{word.lower()}:"
        result = emoji.emojize(shortcode, language="en")
        if result != shortcode:
            return result
    return "🍽"


def is_profile_complete(user_data: dict) -> bool:
    return all(
        [
            user_data.get("weight"),
            user_data.get("height"),
            user_data.get("age"),
            user_data.get("city"),
        ]
    )


async def save_daily_history(user_id: int, user_data: dict):
    today = datetime.now().date().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO history (user_id, date, water_logged, water_goal,
                                calories_logged, calories_burned, calorie_goal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET
                water_logged = excluded.water_logged,
                water_goal = excluded.water_goal,
                calories_logged = excluded.calories_logged,
                calories_burned = excluded.calories_burned,
                calorie_goal = excluded.calorie_goal
            """,
            (
                user_id,
                today,
                user_data.get("logged_water", 0),
                user_data.get("water_goal", 2000),
                user_data.get("logged_calories", 0),
                user_data.get("burned_calories", 0),
                user_data.get("calorie_goal", 2000),
            ),
        )
        await db.commit()


async def get_history(user_id: int, days: int = 7) -> list[dict]:
    start_date = (datetime.now() - timedelta(days=days - 1)).date().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM history
            WHERE user_id = ? AND date >= ?
            ORDER BY date ASC
            """,
            (user_id, start_date),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def generate_progress_chart(user_id: int, days: int = 7) -> Optional[bytes]:
    history = await get_history(user_id, days)
    if not history:
        return None

    dates = [datetime.fromisoformat(h["date"]) for h in history]
    water_logged = [h["water_logged"] for h in history]
    water_goals = [h["water_goal"] for h in history]
    calories_logged = [h["calories_logged"] for h in history]
    calories_burned = [h["calories_burned"] for h in history]
    calorie_goals = [h["calorie_goal"] for h in history]
    calorie_balance = [
        logged - burned for logged, burned in zip(calories_logged, calories_burned)
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    fig.suptitle("Прогресс за последние дни", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.fill_between(dates, water_logged, alpha=0.3, color="#2196F3")
    ax1.plot(
        dates,
        water_logged,
        "o-",
        color="#2196F3",
        linewidth=2,
        markersize=8,
        label="Выпито",
    )
    ax1.plot(dates, water_goals, "--", color="#4CAF50", linewidth=2, label="Цель")
    ax1.set_ylabel("Вода (мл)", fontsize=11)
    ax1.set_title("Потребление воды", fontsize=12, pad=10)
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())

    for i, (d, w, g) in enumerate(zip(dates, water_logged, water_goals)):
        percent = min(100, int(w / g * 100)) if g > 0 else 0
        color = (
            "#4CAF50" if percent >= 100 else "#FF9800" if percent >= 50 else "#F44336"
        )
        ax1.annotate(
            f"{percent}%",
            (d, w),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax2 = axes[1]
    bar_width = 0.35
    x_indices = range(len(dates))

    ax2.bar(
        [i - bar_width / 2 for i in x_indices],
        calories_logged,
        bar_width,
        label="Потреблено",
        color="#FF5722",
        alpha=0.8,
    )
    ax2.bar(
        [i + bar_width / 2 for i in x_indices],
        calories_burned,
        bar_width,
        label="Сожжено",
        color="#4CAF50",
        alpha=0.8,
    )
    ax2.plot(
        x_indices,
        calorie_goals,
        "D--",
        color="#9C27B0",
        linewidth=2,
        markersize=6,
        label="Цель",
    )
    ax2.set_ylabel("Калории (ккал)", fontsize=11)
    ax2.set_title("Калории", fontsize=12, pad=10)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([d.strftime("%d.%m") for d in dates])
    ax2.legend(loc="upper left")

    for i, (bal, goal) in enumerate(zip(calorie_balance, calorie_goals)):
        color = "#4CAF50" if bal <= goal else "#F44336"
        ax2.annotate(
            f"{bal:.0f}",
            (i, max(calories_logged[i], calories_burned[i])),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def calculate_water_goal(
    weight: float, activity: int, temperature: Optional[float] = None
) -> int:
    base = weight * 30
    activity_bonus = (activity // 30) * 500
    weather_bonus = 0
    if temperature is not None:
        if temperature > 30:
            weather_bonus = 1000
        elif temperature > 25:
            weather_bonus = 500
    return int(base + activity_bonus + weather_bonus)


def calculate_calorie_goal(
    weight: float, height: float, age: int, activity: int, gender: str = "male"
) -> int:
    if gender == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    if activity >= 90:
        pal = 1.9
    elif activity >= 60:
        pal = 1.725
    elif activity >= 30:
        pal = 1.55
    elif activity >= 15:
        pal = 1.375
    else:
        pal = 1.2
    return int(bmr * pal)


class ProfileStates(StatesGroup):
    waiting_for_weight = State()
    waiting_for_height = State()
    waiting_for_age = State()
    waiting_for_gender = State()
    waiting_for_activity = State()
    waiting_for_city = State()
    waiting_for_calorie_goal = State()


class FoodStates(StatesGroup):
    waiting_for_grams = State()


@dp.message.middleware()
async def logging_middleware(handler, event: Message, data: dict):
    user = event.from_user
    logger.info(f"User [{user.id}] @{user.username}: {event.text}")
    return await handler(event, data)


async def get_weather(city: str) -> Optional[dict]:
    if not OPENWEATHER_API_KEY:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "ru"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "temp": data["main"]["temp"],
                        "description": data["weather"][0]["description"],
                        "city": data["name"],
                    }
                return None
    except Exception as e:
        logger.error(f"Ошибка при запросе погоды: {e}")
        return None


async def get_food_info(product_name: str) -> Optional[dict]:
    search_term = await translate_to_english(product_name.lower().strip())
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "action": "process",
        "search_terms": search_term,
        "json": "true",
    }
    try:
        connector = ProxyConnector.from_url(PROXY_URL) if PROXY_URL else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    for product in data.get("products", []):
                        nutriments = product.get("nutriments", {})
                        calories = nutriments.get("energy-kcal_100g")
                        if calories and calories > 0:
                            return {
                                "name": product.get("product_name", product_name),
                                "calories": round(calories, 1),
                                "protein": round(nutriments.get("proteins_100g", 0), 1),
                                "fat": round(nutriments.get("fat_100g", 0), 1),
                                "carbs": round(
                                    nutriments.get("carbohydrates_100g", 0), 1
                                ),
                            }
                return None
    except asyncio.TimeoutError:
        logger.error("Таймаут при запросе OpenFoodFacts API")
        return None
    except Exception as e:
        logger.error(f"Ошибка при запросе OpenFoodFacts: {e}")
        return None


async def get_food_calories(
    product_name: str,
) -> tuple[str, Optional[float], Optional[dict]]:
    food_info = await get_food_info(product_name)
    if food_info:
        return (product_name, food_info["calories"], food_info)
    return (product_name, None, None)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        """Привет! Это бот для отслеживания воды, калорий и активности.

*Команды:*
/set\\_profile - задать профиль
/log\\_water <мл> - записать воду
/log\\_food <продукт> - записать еду
/log\\_workout <тип> <мин> - записать тренировку
/check\\_progress - прогресс
/plot - графики прогресса
/recommend - советы

Чтобы начать, задайте ваш профиль: /set\\_profile""",
        parse_mode="Markdown",
    )


@router.message(Command("set_profile"))
async def cmd_set_profile(message: Message, state: FSMContext):
    await state.set_state(ProfileStates.waiting_for_weight)
    await message.answer("Ваш вес (кг):")


@router.message(ProfileStates.waiting_for_weight)
async def process_weight(message: Message, state: FSMContext):
    try:
        weight = float(message.text.replace(",", "."))
        if not 0 < weight <= 500:
            raise ValueError
        await state.update_data(weight=weight)
        await state.set_state(ProfileStates.waiting_for_height)
        await message.answer("Ваш рост (см):")
    except ValueError:
        await message.answer("Введите вес в кг (например, 70):")


@router.message(ProfileStates.waiting_for_height)
async def process_height(message: Message, state: FSMContext):
    try:
        height = float(message.text.replace(",", "."))
        if not 0 < height <= 300:
            raise ValueError
        await state.update_data(height=height)
        await state.set_state(ProfileStates.waiting_for_age)
        await message.answer("Ваш возраст:")
    except ValueError:
        await message.answer("Введите рост в см (например, 175):")


@router.message(ProfileStates.waiting_for_age)
async def process_age(message: Message, state: FSMContext):
    try:
        age = int(message.text)
        if not 0 < age <= 150:
            raise ValueError
        await state.update_data(age=age)
        await state.set_state(ProfileStates.waiting_for_gender)
        await message.answer("Ваш пол (М/Ж):")
    except ValueError:
        await message.answer("Введите возраст (например, 25):")


@router.message(ProfileStates.waiting_for_gender)
async def process_gender(message: Message, state: FSMContext):
    text = message.text.lower().strip()
    if text in ["м", "m", "мужской", "male", "муж"]:
        gender = "male"
    elif text in ["ж", "f", "женский", "female", "жен"]:
        gender = "female"
    else:
        await message.answer("Введите М или Ж:")
        return
    await state.update_data(gender=gender)
    await state.set_state(ProfileStates.waiting_for_activity)
    await message.answer("Ваши минуты активности в день:")


@router.message(ProfileStates.waiting_for_activity)
async def process_activity(message: Message, state: FSMContext):
    try:
        activity = int(message.text)
        if not 0 <= activity <= 1440:
            raise ValueError
        await state.update_data(activity=activity)
        await state.set_state(ProfileStates.waiting_for_city)
        await message.answer("Ваш город (на английском, например: Moscow):")
    except ValueError:
        await message.answer("Введите минуты (например, 30):")


@router.message(ProfileStates.waiting_for_city)
async def process_city(message: Message, state: FSMContext):
    city = message.text.strip()
    weather = await get_weather(city)
    if weather:
        await state.update_data(city=city, temperature=weather["temp"])
        await state.set_state(ProfileStates.waiting_for_calorie_goal)
        data = await state.get_data()
        calorie_goal = calculate_calorie_goal(
            data["weight"],
            data["height"],
            data["age"],
            data["activity"],
            data.get("gender", "male"),
        )
        await message.answer(
            f"""Погода в {weather['city']}: {weather['temp']}°C, {weather['description']}

Рекомендуемая норма калорий: {calorie_goal} ккал

Ваша цель по калориям (или 'авто'):"""
        )
    else:
        await message.answer("Город не найден. Попробуйте другой:")


@router.message(ProfileStates.waiting_for_calorie_goal)
async def process_calorie_goal(message: Message, state: FSMContext):
    data = await state.get_data()
    if message.text.lower() in ["авто", "auto", "автоматически"]:
        calorie_goal = calculate_calorie_goal(
            data["weight"],
            data["height"],
            data["age"],
            data["activity"],
            data.get("gender", "male"),
        )
    else:
        try:
            calorie_goal = int(message.text)
            if not 0 < calorie_goal <= 10000:
                raise ValueError
        except ValueError:
            await message.answer("Введите число или 'авто':")
            return

    water_goal = calculate_water_goal(
        data["weight"], data["activity"], data.get("temperature")
    )
    user_id = message.from_user.id

    await update_user_data(
        user_id,
        weight=data["weight"],
        height=data["height"],
        age=data["age"],
        gender=data.get("gender", "male"),
        activity=data["activity"],
        city=data["city"],
        water_goal=water_goal,
        calorie_goal=calorie_goal,
    )
    await state.clear()

    gender_text = "М" if data.get("gender") == "male" else "Ж"
    await message.answer(
        f"""Профиль сохранён!

Вес: {data['weight']} кг, Рост: {data['height']} см
Возраст: {data['age']}, Пол: {gender_text}
Активность: {data['activity']} мин/день
Город: {data['city']}

Цели: вода {water_goal} мл, калории {calorie_goal} ккал
Для просмотра прогресса используйте команду /check_progress"""
    )


@router.message(Command("log_water"))
async def cmd_log_water(message: Message):
    user_id = message.from_user.id
    user_data = await get_user_data(user_id)

    if not is_profile_complete(user_data):
        await message.answer("Сначала настройте профиль: /set_profile")
        return

    args = message.text.split()
    if len(args) < 2:
        await message.answer("Формат: /log_water <мл>\nПример: /log_water 250")
        return

    try:
        amount = int(args[1])
        if not 0 < amount <= 5000:
            raise ValueError
    except ValueError:
        await message.answer("Введите 1-5000 мл")
        return

    new_water = user_data["logged_water"] + amount
    await update_user_data(user_id, logged_water=new_water)
    user_data["logged_water"] = new_water
    await save_daily_history(user_id, user_data)

    remaining = max(0, user_data["water_goal"] - new_water)
    status = "Норма выполнена!" if remaining == 0 else f"Осталось: {remaining} мл"
    progress_percent = min(100, int(new_water / user_data["water_goal"] * 100))
    progress_bar = "█" * (progress_percent // 10) + "░" * (10 - progress_percent // 10)

    await message.answer(
        f"""+{amount} мл воды

Вода: {new_water}/{user_data['water_goal']} мл
[{progress_bar}] {progress_percent}%
{status}"""
    )


@router.message(Command("log_food"))
async def cmd_log_food(message: Message, state: FSMContext):
    user_id = message.from_user.id
    user_data = await get_user_data(user_id)

    if not is_profile_complete(user_data):
        await message.answer("Сначала настройте профиль: /set_profile")
        return

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("Формат: /log_food <продукт>\nПример: /log_food банан")
        return

    product_name = args[1]
    name, calories, food_info = await get_food_calories(product_name)

    if calories is None:
        await message.answer(
            f'Продукт "{product_name}" не найден.\n'
            "Попробуйте на английском или другое название."
        )
        return

    await state.set_state(FoodStates.waiting_for_grams)
    await state.update_data(food_name=name, food_calories=calories, food_info=food_info)

    food_emoji = await get_food_emoji(name)
    info_lines = [f"{food_emoji} {name} - {calories} ккал/100г"]

    if food_info and food_info.get("protein"):
        info_lines.append(
            f"   Б: {food_info['protein']}г | Ж: {food_info['fat']}г | У: {food_info['carbs']}г"
        )
    info_lines.append("\nСколько грамм?")

    await message.answer("\n".join(info_lines))


@router.message(FoodStates.waiting_for_grams)
async def process_food_grams(message: Message, state: FSMContext):
    try:
        grams = float(message.text.replace(",", "."))
        if not 0 < grams <= 10000:
            raise ValueError
    except ValueError:
        await message.answer("Введите количество грамм:")
        return

    data = await state.get_data()
    calories = data["food_calories"] * grams / 100

    user_id = message.from_user.id
    user_data = await get_user_data(user_id)
    new_calories = user_data["logged_calories"] + calories
    await update_user_data(user_id, logged_calories=new_calories)
    user_data["logged_calories"] = new_calories
    await save_daily_history(user_id, user_data)
    await state.clear()

    remaining = max(
        0, user_data["calorie_goal"] - new_calories + user_data["burned_calories"]
    )
    await message.answer(
        f"""+{calories:.0f} ккал ({data['food_name']}, {grams:.0f}г)

Потреблено: {new_calories:.0f} ккал
Сожжено: {user_data['burned_calories']:.0f} ккал
Осталось: {remaining:.0f} ккал"""
    )


@router.message(Command("log_workout"))
async def cmd_log_workout(message: Message):
    user_id = message.from_user.id
    user_data = await get_user_data(user_id)

    if not is_profile_complete(user_data):
        await message.answer("Сначала настройте профиль: /set_profile")
        return

    args = message.text.split()
    if len(args) < 3:
        workout_types = ", ".join(WORKOUTS.keys())
        await message.answer(
            f"""Формат: /log_workout <тип> <мин>
Пример: /log_workout бег 30

Типы: {workout_types}"""
        )
        return

    workout_input = args[1].lower()
    try:
        duration = int(args[2])
        if not 0 < duration <= 600:
            raise ValueError
    except ValueError:
        await message.answer("Введите 1-600 минут")
        return

    workout_name, workout_data = get_workout_data(workout_input)
    workout_emoji = workout_data["emoji"]
    weight = user_data["weight"]

    if workout_name in WORKOUTS:
        cals_per_min = calories_per_min(workout_name, weight)
    else:
        cals_per_min = workout_data["met"] * weight * KCAL_PER_MIN_PER_KG

    calories = cals_per_min * duration
    extra_water = (duration // 30) * 200 + (200 if duration % 30 >= 15 else 0)

    new_burned = user_data["burned_calories"] + calories
    await update_user_data(user_id, burned_calories=new_burned)
    user_data["burned_calories"] = new_burned
    await save_daily_history(user_id, user_data)

    await message.answer(
        f"""{workout_emoji} {workout_name.capitalize()} {duration} мин - {calories:.0f} ккал

Выпейте {extra_water} мл воды
Всего сожжено: {new_burned:.0f} ккал"""
    )


@router.message(Command("check_progress"))
async def cmd_check_progress(message: Message):
    user_id = message.from_user.id
    user_data = await get_user_data(user_id)

    if not is_profile_complete(user_data):
        await message.answer("Сначала настройте профиль: /set_profile")
        return

    weather = await get_weather(user_data["city"])
    weather_info = ""
    if weather:
        new_water_goal = calculate_water_goal(
            user_data["weight"], user_data["activity"], weather["temp"]
        )
        await update_user_data(user_id, water_goal=new_water_goal)
        user_data["water_goal"] = new_water_goal
        weather_info = (
            f"{weather['city']}: {weather['temp']}°C, {weather['description']}\n\n"
        )

    water_consumed = user_data["logged_water"]
    water_goal = user_data["water_goal"]
    water_remaining = max(0, water_goal - water_consumed)
    water_percent = min(100, int(water_consumed / water_goal * 100))
    water_bar = "█" * (water_percent // 10) + "░" * (10 - water_percent // 10)

    calories_consumed = user_data["logged_calories"]
    calories_burned = user_data["burned_calories"]
    calorie_goal = user_data["calorie_goal"]
    calorie_balance = calories_consumed - calories_burned
    calories_remaining = max(0, calorie_goal - calorie_balance)
    calorie_percent = (
        min(100, int(calorie_balance / calorie_goal * 100)) if calorie_goal > 0 else 0
    )
    calorie_bar = "█" * (calorie_percent // 10) + "░" * (10 - calorie_percent // 10)

    await message.answer(
        f"""*Прогресс*

{weather_info}*Вода:*
Выпито: {water_consumed}/{water_goal} мл
[{water_bar}] {water_percent}%
Осталось: {water_remaining} мл

*Калории:*
Потреблено: {calories_consumed:.0f}/{calorie_goal} ккал
Сожжено: {calories_burned:.0f} ккал
[{calorie_bar}] {calorie_percent}%
Баланс: {calorie_balance:.0f}, осталось: {calories_remaining:.0f} ккал""",
        parse_mode="Markdown",
    )


@router.message(Command("recommend"))
async def cmd_recommend(message: Message):
    user_id = message.from_user.id
    user_data = await get_user_data(user_id)

    if not is_profile_complete(user_data):
        await message.answer("Сначала настройте профиль: /set_profile")
        return

    recommendations = ["*Рекомендации:*\n"]
    water_percent = (user_data["logged_water"] / user_data["water_goal"]) * 100

    if water_percent < 30:
        recommendations.append("Мало воды! Пейте по 250 мл каждый час.")
    elif water_percent < 60:
        recommendations.append("Хороший темп, продолжайте.")
    else:
        recommendations.append("Норма воды выполнена!")

    calorie_balance = user_data["logged_calories"] - user_data["burned_calories"]
    calories_remaining = user_data["calorie_goal"] - calorie_balance

    if calories_remaining > 500:
        recommendations.extend(
            [
                "\n*Низкокалорийные продукты:*",
                "• Салат (50), огурец (15), помидор (18) ккал/100г",
                "• Яблоко (52 ккал/100г)",
            ]
        )
    elif calories_remaining < 0:
        extra_minutes = int(abs(calories_remaining) / 10)
        recommendations.extend(
            [
                f"\nПревышение на {abs(calories_remaining):.0f} ккал!",
                f"\n*Тренировки:*",
                f"• Бег {extra_minutes} мин (~{extra_minutes * 10} ккал)",
                f"• Ходьба {extra_minutes * 2} мин",
            ]
        )

    if user_data["burned_calories"] == 0:
        weight = user_data["weight"] or 70
        recommendations.extend(
            [
                "\nСегодня без тренировок. Попробуйте:",
                f"• Ходьба 30 мин = ~{int(calories_per_min('ходьба', weight) * 30)} ккал",
                f"• Бег 20 мин = ~{int(calories_per_min('бег', weight) * 20)} ккал",
            ]
        )

    await message.answer("\n".join(recommendations), parse_mode="Markdown")


@router.message(Command("plot"))
async def cmd_plot(message: Message):
    user_id = message.from_user.id
    user_data = await get_user_data(user_id)

    if not is_profile_complete(user_data):
        await message.answer("Сначала настройте профиль: /set_profile")
        return

    await save_daily_history(user_id, user_data)

    args = message.text.split()
    days = 7
    if len(args) > 1:
        try:
            days = int(args[1])
            if not 1 <= days <= 30:
                days = 7
        except ValueError:
            pass

    await message.answer("Генерирую график...")

    chart_data = await generate_progress_chart(user_id, days)

    if chart_data is None:
        await message.answer(
            "Нет данных для построения графика.\n"
            "Записывайте воду, еду и тренировки, чтобы увидеть прогресс!"
        )
        return

    photo = BufferedInputFile(chart_data, filename="progress.png")
    await message.answer_photo(
        photo,
        caption=f"Ваш прогресс за последние {days} дней\n\n"
        f"Вода и Калории\n"
        f"Используйте /plot <дни> для другого периода (1-30)",
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        """*Справка:*
/set\\_profile - задать профиль
/log\\_water <мл> - записать воду
/log\\_food <продукт> - записать еду
/log\\_workout <тип> <мин> - записать тренировку
/check\\_progress - прогресс
/plot - графики прогресса
/recommend - советы

*Примеры:*
/log\\_water 250
/log\\_food банан
/log\\_workout бег 30
/plot 14""",
        parse_mode="Markdown",
    )


@router.message()
async def unknown_message(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Неизвестная команда. /help")


async def main():
    await init_db()
    dp.include_router(router)

    commands = [
        BotCommand(command="start", description="Начать работу с ботом"),
        BotCommand(command="set_profile", description="Настроить профиль"),
        BotCommand(command="log_water", description="Записать выпитую воду"),
        BotCommand(command="log_food", description="Записать съеденную еду"),
        BotCommand(command="log_workout", description="Записать тренировку"),
        BotCommand(command="check_progress", description="Посмотреть прогресс"),
        BotCommand(command="plot", description="Графики прогресса"),
        BotCommand(command="recommend", description="Получить рекомендации"),
        BotCommand(command="help", description="Справка по командам"),
    ]
    await bot.set_my_commands(commands)
    logger.info("Бот запущен!")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
