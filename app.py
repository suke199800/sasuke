from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import logging
import json # JSON 처리를 위해 추가
from datetime import datetime # 타임스탬프를 위해 추가
import pytz # 시간대 처리를 위해 추가 (pip install pytz 필요)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- 상수 정의 ---
SYSTEM_PROMPT = """당신은 친철한 사람입니다. 말투는 사극말투로 단아하게 말해주세요."""

GUESTBOOK_FILE = 'guestbook.json' # 방명록 데이터 파일 경로
TIMEZONE = pytz.timezone('Asia/Seoul') # 한국 시간대 설정

# --- Gemini API 설정 ---
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

# --- 방명록 데이터 처리 함수 ---
def load_guestbook_entries():
    """guestbook.json 파일에서 방명록 데이터를 로드합니다."""
    try:
        # 파일이 UTF-8 인코딩으로 저장되도록 명시
        with open(GUESTBOOK_FILE, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            # 데이터 형식 검증 (리스트인지 확인)
            if not isinstance(entries, list):
                app.logger.warning(f"{GUESTBOOK_FILE} 내용이 리스트 형식이 아닙니다. 빈 리스트를 반환합니다.")
                return []
            return entries
    except FileNotFoundError:
        app.logger.info(f"{GUESTBOOK_FILE} 파일을 찾을 수 없어 빈 리스트를 반환합니다.")
        return []
    except json.JSONDecodeError:
        app.logger.error(f"{GUESTBOOK_FILE} 파일의 JSON 형식이 잘못되었습니다. 빈 리스트를 반환합니다.")
        return []
    except Exception as e:
        app.logger.error(f"방명록 로딩 중 오류 발생: {e}")
        return []

def save_guestbook_entries(entries):
    """방명록 데이터를 guestbook.json 파일에 저장합니다."""
    try:
        # 파일이 UTF-8 인코딩으로 저장되도록 명시, ensure_ascii=False로 한글 유지
        with open(GUESTBOOK_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)
        app.logger.info(f"방명록이 {GUESTBOOK_FILE}에 성공적으로 저장되었습니다.")
        return True
    except Exception as e:
        app.logger.error(f"방명록 저장 중 오류 발생: {e}")
        return False

# --- Flask 라우트 ---
@app.route('/')
def index():
    """메인 페이지를 렌더링하고 초기 방명록 데이터를 전달합니다."""
    try:
        # 페이지 로드 시 방명록 데이터 로드
        current_entries = load_guestbook_entries()
        # 최신 글이 위로 오도록 순서 뒤집기 (선택 사항)
        current_entries.reverse()
        return render_template('index.html', guestbook_entries=current_entries)
    except Exception as e:
        app.logger.error(f"index.html 렌더링 중 오류: {e}")
        return "HTML 페이지를 로드하는 중 오류가 발생했습니다.", 500

# '/ask' 라우트는 이전과 동일하게 유지 (시스템 프롬프트 추가된 버전)
@app.route('/ask', methods=['POST'])
def ask_gemini():
    global model
    if not model:
        app.logger.error("API 요청 수신 실패: Gemini 모델이 로드되지 않음")
        return jsonify({"error": "Gemini 모델을 초기화하지 못했습니다."}), 500

    data = request.get_json()
    if not data or 'history' not in data:
        return jsonify({"error": "잘못된 요청 형식입니다. 'history' 키가 필요합니다."}), 400

    conversation_history = data.get('history')
    if not conversation_history or not isinstance(conversation_history, list):
        return jsonify({"error": "'history'는 비어 있지 않은 리스트여야 합니다."}), 400
    if not conversation_history[-1].get('role') == 'user':
         return jsonify({"error": "마지막 메시지는 사용자 질문이어야 합니다."}), 400

    gemini_formatted_user_history = []
    for message in conversation_history:
        role = message.get('role')
        content = message.get('content')
        if not role or content is None:
            continue
        gemini_role = 'model' if role == 'assistant' else role
        gemini_formatted_user_history.append({'role': gemini_role, 'parts': [content]})

    if not gemini_formatted_user_history:
         return jsonify({"error": "처리할 유효한 대화 내용이 없습니다."}), 400

    initial_context = [
        {'role': 'user', 'parts': [SYSTEM_PROMPT]},
        {'role': 'model', 'parts': ["네, 알겠습니다. END 길드 비서로서 무엇을 도와드릴까요?"]}
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
            app.logger.warning(f"Gemini 응답 비어있음: {response}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                answer_text = f"죄송합니다. 답변 불가 (사유: {reason})"
            else:
                answer_text = "죄송합니다. 답변 생성 중 문제 발생."

        return jsonify({"answer": answer_text})

    except Exception as e:
        app.logger.error(f"Gemini API 호출 중 오류: {e}", exc_info=True)
        return jsonify({"error": f"Gemini API 통신 오류."}), 500

# --- 방명록 API 엔드포인트 ---
@app.route('/guestbook', methods=['GET'])
def get_guestbook():
    """현재 방명록 목록을 반환합니다."""
    entries = load_guestbook_entries()
    # 최신 글이 위로 오도록 순서 뒤집기
    entries.reverse()
    return jsonify(entries)

@app.route('/guestbook', methods=['POST'])
def add_guestbook_entry():
    """새 방명록 글을 추가합니다."""
    data = request.get_json()
    name = data.get('name', '').strip()
    message = data.get('message', '').strip()

    if not name or not message:
        return jsonify({"error": "이름과 메시지를 모두 입력해주세요."}), 400
    # 간단한 길이 제한 (선택 사항)
    if len(name) > 20 or len(message) > 500:
        return jsonify({"error": "이름은 20자, 메시지는 500자 이하로 입력해주세요."}), 400

    # 현재 시간 (한국 시간대 기준)
    timestamp = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')

    new_entry = {
        "name": name,
        "message": message,
        "timestamp": timestamp
    }

    entries = load_guestbook_entries()
    entries.append(new_entry) # 새 글을 리스트 끝에 추가

    if save_guestbook_entries(entries):
        # 성공 시 추가된 글 정보 반환 (선택 사항)
        return jsonify({"success": True, "entry": new_entry}), 201
    else:
        return jsonify({"error": "방명록 저장 중 서버 오류가 발생했습니다."}), 500

# --- 서버 실행 ---
if __name__ == '__main__':
    # requirements.txt에 pytz 추가하는 것을 잊지 마세요!
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
