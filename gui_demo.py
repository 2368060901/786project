import streamlit as st
import rlcard
from rlcard.agents.random_agent import RandomAgent
from PIL import Image
import os

def convert_card(card_str):
    rank_map = {'A': 'ace', 'K': 'king', 'Q': 'queen', 'J': 'jack', 'T': '10',
                '9': '9', '8': '8', '7': '7', '6': '6', '5': '5', '4': '4', '3': '3', '2': '2', '10': '10'}
    suit_map = {'S': 'spades', 'H': 'hearts', 'D': 'diamonds', 'C': 'clubs'}

    rank = card_str[-1]  # All but last char
    suit = card_str[0]   # Last char

    rank_full = rank_map.get(rank)
    suit_full = suit_map.get(suit)


    if rank_full and suit_full:
        return f"{rank_full}_of_{suit_full}"


# Load card images from local folder
def load_card_image(card_str):
    filename = convert_card(card_str)
    front_path = os.path.join("cards", filename + ".png") if filename else None
    back_path = os.path.join("cards", "back.png")

    if not os.path.exists(back_path):
        raise FileNotFoundError("Missing cards/back.png")

    back = Image.open(back_path).convert("RGBA")
    back = back.resize((back.width, back.height))

    if front_path and os.path.exists(front_path):
        front = Image.open(front_path).convert("RGBA")
        front = front.resize(back.size)
        back.paste(front, (0, 0), front)

    return back

# Setup environment
env = rlcard.make('no-limit-holdem')
env.set_agents([None, RandomAgent(num_actions=env.num_actions)])

# Initialize session state
if 'state' not in st.session_state:
    state, _ = env.reset()
    st.session_state.state = state
    st.session_state.env = env
    st.session_state.episode_over = False
    st.session_state.rewards = None
    st.session_state.current_player = 0

st.markdown(
    """
    <style>
    body {
        background-color: #0a472e; /* 类似绿色卡桌背景 */
        color: white;
    }
    .stApp {
        background-color: #0a472e;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Texas Hold 'em online (Streamlit GUI)")

state = st.session_state.state
raw_obs = state['raw_obs']

st.subheader("Your hand Card:")
cols = st.columns(2)
for i, card in enumerate(raw_obs['hand']):
    img = load_card_image(card)
    if img:
        cols[i].image(img, width=100)
    else:
        cols[i].write(card)

st.subheader("Community Cards")
public_cols = st.columns(5)
for i in range(5):
    if i < len(raw_obs['public_cards']):
        img = load_card_image(raw_obs['public_cards'][i])
        if img:
            public_cols[i].image(img, width=100)
        else:
            public_cols[i].write(raw_obs['public_cards'][i])
    else:
        public_cols[i].image("cards/back.png", width=100)

st.write("Current chip:", raw_obs['my_chips'], "| Current pool:", raw_obs['pot'])

# 显示动作按钮
legal_actions = list(state['legal_actions'].keys())
action_names = state['raw_legal_actions']
action_map = dict(zip(legal_actions, action_names))

chosen_action = st.radio("Choose your action:", options=legal_actions, format_func=lambda x: str(action_map[x]))

if st.button("Confirm your action"):
    next_state, next_player = st.session_state.env.step(chosen_action)
    st.session_state.state = next_state
    st.session_state.current_player = next_player

    # 检查游戏是否结束
    if next_player == -1:
        _, payoffs = st.session_state.env.run(is_training=False)
        st.session_state.episode_over = True
        st.session_state.rewards = payoffs

# 如果游戏结束
if st.session_state.episode_over:
    st.success("游戏结束！")
    st.write("你的收益：", st.session_state.rewards[0])
    if st.button("重新开始"):
        state, _ = env.reset()
        st.session_state.state = state
        st.session_state.episode_over = False
        st.session_state.rewards = None
        st.session_state.current_player = 0