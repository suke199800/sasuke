import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import google.generativeai as genai
import os
import logging
import json
from datetime import datetime
import pytz
import psycopg2
from psycopg2 import sql
from urllib.parse import urlparse

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """당신은 친철한 사람입니다. 말투는 사극말투로 단아하게 말해주세요."""
TIMEZONE = pytz.timezone('Asia/Seoul')
DATABASE_URL = os.environ.get('DATABASE_URL')

model = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    app.logger.info("Gemini 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    app.logger.error(f"Gemini API 설정 중 오류 발생: {e}")

def get_db_connection():
    if not DATABASE_URL:
        app.logger.error("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        app.logger.error(f"데이터베이스 연결 실패: {e}")
        return None

def initialize_database():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS guestbook (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            app.logger.info("데이터베이스 테이블이 준비되었습니다.")
        except Exception as e:
            app.logger.error(f"데이터베이스 테이블 생성 중 오류: {e}")
        finally:
            conn.close()
    else:
        app.logger.error("데이터베이스 연결 실패로 테이블 초기화 건너뜀.")

def load_guestbook_entries_from_db():
    entries = []
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT name, message, timestamp FROM guestbook ORDER BY timestamp DESC")
                rows = cur.fetchall()
                for row in rows:
                    formatted_timestamp = row[2].astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                    entries.append({
                        "name": row[0],
                        "message": row[1],
                        "timestamp": formatted_timestamp
                    })
        except Exception as e:
            app.logger.error(f"DB에서 방명록 로딩 중 오류 발생: {e}")
        finally:
            conn.close()
    return entries

@app.route('/')
def index():
    try:
        current_entries = load_guestbook_entries_from_db()
        return render_template('index.html', guestbook_entries=current_entries)
    except Exception as e:
        app.logger.error(f"index.html 렌더링 중 오류: {e}")
        return "HTML 페이지를 로드하는 중 오류가 발생했습니다.", 500

@app.route('/ask', methods=['POST'])
def ask_gemini():
    global model
    if not model:
        return jsonify({"error": "Gemini 모델을 초기화하지 못했습니다."}), 500

    data = request.get_json()
    if not data or 'history' not in data:
        return jsonify({"error": "잘못된 요청 형식: 'history' 필요"}), 400

    conversation_history = data.get('history')
    if not conversation_history or not isinstance(conversation_history, list):
         return jsonify({"error": "'history'는 비어 있지 않은 리스트여야 합니다."}), 400
    if not conversation_history[-1].get('role') == 'user':
         return jsonify({"error": "마지막 메시지는 사용자 질문이어야 합니다."}), 400

    gemini_formatted_user_history = []
    for message in conversation_history:
        role = message.get('role')
        content = message.get('content')
        if not role or content is None: continue
        gemini_role = 'model' if role == 'assistant' else role
        gemini_formatted_user_history.append({'role': gemini_role, 'parts': [content]})

    if not gemini_formatted_user_history:
         return jsonify({"error": "처리할 유효한 대화 내용 없음"}), 400

    initial_context = [
        {'role': 'user', 'parts': [SYSTEM_PROMPT]},
        {'role': 'model', 'parts': ["네, 그리 하겠사옵니다. 무엇을 여쭈시려 하시는지요?"]}
    ]
    full_history_for_api = initial_context + gemini_formatted_user_history

    app.logger.info(f"Gemini에게 전달할 총 history 개수: {len(full_history_for_api)}")

    try:
        response = model.generate_content(full_history_for_api)
        answer_text = ""
        if hasattr(response, 'parts') and response.parts:
            answer_text = "".join(part.text for part in response.parts)
        elif hasattr(response, 'text'):
             answer_text = response.text

        if not answer_text:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                answer_text = f"송구하오나, 답변드릴 수 없사옵니다. (사유: {reason})"
            else:
                answer_text = "송구하오나, 답변을 마련하는 중에 문제가 생겼사옵니다."

        return jsonify({"answer": answer_text})

    except Exception as e:
        app.logger.error(f"Gemini API 호출 중 오류: {e}", exc_info=True)
        return jsonify({"error": f"API와 연결하는 중에 문제가 발생하였사옵니다."}), 500

@app.route('/guestbook', methods=['GET'])
def get_guestbook():
    entries = load_guestbook_entries_from_db()
    return jsonify(entries)

@app.route('/guestbook', methods=['POST'])
def add_guestbook_entry():
    data = request.get_json()
    name = data.get('name', '').strip()
    message = data.get('message', '').strip()

    if not name or not message:
        return jsonify({"error": "성함과 남기실 말씀을 모두 적어주시옵소서."}), 400
    if len(name) > 50 or len(message) > 1000:
        return jsonify({"error": "글자 수 제한을 확인해주시옵소서."}), 400

    conn = get_db_connection()
    new_entry_data = None
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO guestbook (name, message) VALUES (%s, %s) RETURNING name, message, timestamp",
                    (name, message)
                )
                inserted_row = cur.fetchone()
                conn.commit()

                if inserted_row:
                    formatted_timestamp = inserted_row[2].astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                    new_entry_data = {
                        "name": inserted_row[0],
                        "message": inserted_row[1],
                        "timestamp": formatted_timestamp
                    }
                    app.logger.info(f"새 방명록 등록: {new_entry_data}")
                    socketio.emit('new_entry', new_entry_data)

        except Exception as e:
            conn.rollback()
            app.logger.error(f"DB에 방명록 저장 중 오류: {e}")
            return jsonify({"error": "방명록 저장 중 서버 오류 발생."}), 500
        finally:
            conn.close()

        if new_entry_data:
            return jsonify({"success": True, "entry": new_entry_data}), 201
        else:
             return jsonify({"error": "방명록 저장 확인 중 문제 발생."}), 500
    else:
        return jsonify({"error": "데이터베이스 연결 실패."}), 500

if __name__ == '__main__':
    with app.app_context():
         initialize_database()

    port = int(os.environ.get('PORT', 5000))
    app.logger.info(f"SocketIO 서버를 {port} 포트에서 시작합니다. (eventlet)")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
