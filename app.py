from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit # SocketIO 추가
import google.generativeai as genai
import os
import logging
import json
from datetime import datetime
import pytz
import psycopg2 # PostgreSQL 드라이버 추가
from psycopg2 import sql
from urllib.parse import urlparse # Database URL 파싱용

# --- 기본 설정 ---
app = Flask(__name__)
# SocketIO 설정: async_mode='eventlet' 또는 'gevent' 사용
# requirements.txt 및 시작 명령어와 일치시켜야 함
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO)

# --- 상수 및 환경 변수 ---
SYSTEM_PROMPT = """당신은 친철한 사람입니다. 말투는 사극말투로 단아하게 말해주세요."""
TIMEZONE = pytz.timezone('Asia/Seoul')
DATABASE_URL = os.environ.get('DATABASE_URL') # 환경 변수에서 DB URL 로드

# --- DB 연결 함수 ---
def get_db_connection():
    """데이터베이스 연결을 생성하고 반환합니다."""
    if not DATABASE_URL:
        app.logger.error("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
        return None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        app.logger.error(f"데이터베이스 연결 실패: {e}")
        return None

# --- DB 테이블 초기화 함수 ---
def initialize_database():
    """애플리케이션 시작 시 데이터베이스 테이블을 생성합니다."""
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

# --- Gemini API 설정 ---
model = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17') # 최신 모델 권장
    app.logger.info("Gemini 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    app.logger.error(f"Gemini API 설정 중 오류 발생: {e}")

# --- 데이터베이스에서 방명록 로드 함수 ---
def load_guestbook_entries_from_db():
    """데이터베이스에서 방명록 데이터를 로드합니다."""
    entries = []
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT name, message, timestamp FROM guestbook ORDER BY timestamp DESC")
                rows = cur.fetchall()
                for row in rows:
                    # 시간대를 한국 시간으로 변환하여 보기 좋게 포맷
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

# --- Flask 라우트 ---
@app.route('/')
def index():
    """메인 페이지 렌더링 (초기 방명록 데이터 로드)"""
    try:
        current_entries = load_guestbook_entries_from_db()
        # DB에서 이미 DESC로 정렬했으므로 reverse 필요 없음
        return render_template('index.html', guestbook_entries=current_entries)
    except Exception as e:
        app.logger.error(f"index.html 렌더링 중 오류: {e}")
        return "HTML 페이지를 로드하는 중 오류가 발생했습니다.", 500

# '/ask' 라우트는 이전과 동일
@app.route('/ask', methods=['POST'])
def ask_gemini():
    global model
    if not model:
        return jsonify({"error": "Gemini 모델을 초기화하지 못했습니다."}), 500

    data = request.get_json()
    if not data or 'history' not in data:
        return jsonify({"error": "잘못된 요청 형식: 'history' 필요"}), 400

    conversation_history = data.get('history')
    # ... (history 유효성 검사 및 Gemini 포맷 변환 로직 - 이전 코드와 동일) ...
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
        # ... (응답 처리 로직 - 이전 코드와 동일) ...
        answer_text = ""
        if hasattr(response, 'parts') and response.parts:
            answer_text = "".join(part.text for part in response.parts)
        # ... (빈 응답, 오류 처리 - 이전 코드와 동일, 말투 수정됨) ...
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

# --- 방명록 API (DB 사용 및 SocketIO 연동) ---
@app.route('/guestbook', methods=['GET'])
def get_guestbook():
    """현재 방명록 목록을 DB에서 로드하여 반환"""
    entries = load_guestbook_entries_from_db()
    return jsonify(entries)

@app.route('/guestbook', methods=['POST'])
def add_guestbook_entry():
    """새 방명록 글을 DB에 추가하고 SocketIO로 브로드캐스트"""
    data = request.get_json()
    name = data.get('name', '').strip()
    message = data.get('message', '').strip()

    # 입력값 검증
    if not name or not message:
        return jsonify({"error": "성함과 남기실 말씀을 모두 적어주시옵소서."}), 400
    if len(name) > 50 or len(message) > 1000: # DB 컬럼 크기에 맞춰 조정 가능
        return jsonify({"error": "글자 수 제한을 확인해주시옵소서."}), 400

    conn = get_db_connection()
    new_entry_data = None
    if conn:
        try:
            with conn.cursor() as cur:
                # 파라미터화된 쿼리로 SQL 인젝션 방지
                cur.execute(
                    "INSERT INTO guestbook (name, message) VALUES (%s, %s) RETURNING name, message, timestamp",
                    (name, message)
                )
                inserted_row = cur.fetchone()
                conn.commit()

                if inserted_row:
                    # 시간 포맷팅
                    formatted_timestamp = inserted_row[2].astimezone(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                    new_entry_data = {
                        "name": inserted_row[0],
                        "message": inserted_row[1],
                        "timestamp": formatted_timestamp
                    }
                    app.logger.info(f"새 방명록 등록: {new_entry_data}")
                    # SocketIO 이벤트 발생: 모든 클라이언트에게 새 글 데이터 전송
                    socketio.emit('new_entry', new_entry_data)

        except Exception as e:
            conn.rollback() # 오류 발생 시 롤백
            app.logger.error(f"DB에 방명록 저장 중 오류: {e}")
            return jsonify({"error": "방명록 저장 중 서버 오류 발생."}), 500
        finally:
            conn.close()

        if new_entry_data:
            return jsonify({"success": True, "entry": new_entry_data}), 201
        else:
             # 삽입은 성공했으나 RETURNING 실패 등 예외 케이스
             return jsonify({"error": "방명록 저장 확인 중 문제 발생."}), 500
    else:
        return jsonify({"error": "데이터베이스 연결 실패."}), 500

# --- 애플리케이션 시작 ---
if __name__ == '__main__':
    initialize_database() # 앱 시작 시 DB 테이블 확인/생성
    port = int(os.environ.get('PORT', 5000))
    # Flask 앱 대신 SocketIO 앱 실행
    app.logger.info(f"SocketIO 서버를 {port} 포트에서 시작합니다.")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
    # Gunicorn 사용 시: gunicorn --worker-class eventlet -w 1 app:app
