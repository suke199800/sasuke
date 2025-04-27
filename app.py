from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import logging
import json
from datetime import datetime
import pytz

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """당신은 친철한 사람입니다. 말투는 사극말투로 단아하게 말해주세요."""

GUESTBOOK_FILE = 'guestbook.json'
TIMEZONE = pytz.timezone('Asia/Seoul')

model = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("환경 변수에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    genai.configure(api_key=api_key)
    # 이전 코드에서 사용된 모델명을 유지합니다. 필요시 gemini-1.5-flash-latest 등으로 변경하세요.
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
    app.logger.info("Gemini 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    app.logger.error(f"Gemini API 설정 중 오류 발생: {e}")


def load_guestbook_entries():
    try:
        with open(GUESTBOOK_FILE, 'r', encoding='utf-8') as f:
            entries = json.load(f)
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
    try:
        with open(GUESTBOOK_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)
        app.logger.info(f"방명록이 {GUESTBOOK_FILE}에 성공적으로 저장되었습니다.")
        return True
    except Exception as e:
        app.logger.error(f"방명록 저장 중 오류 발생: {e}")
        return False


@app.route('/')
def index():
    try:
        current_entries = load_guestbook_entries()
        current_entries.reverse()
        return render_template('index.html', guestbook_entries=current_entries)
    except Exception as e:
        app.logger.error(f"index.html 렌더링 중 오류: {e}")
        return "HTML 페이지를 로드하는 중 오류가 발생했습니다.", 500


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

    # 시스템 프롬프트 정의 부분을 수정하여 사용자 정의 응답을 추가합니다.
    # AI가 프롬프트를 인지하고 역할을 받아들였다는 첫 응답을 설정합니다.
    initial_context = [
        {'role': 'user', 'parts': [SYSTEM_PROMPT]},
        {'role': 'model', 'parts': ["네, 그리 하겠사옵니다. 무엇을 여쭈시려 하시는지요?"]} # 수정된 초기 응답
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
                answer_text = f"송구하오나, 답변드릴 수 없사옵니다. (사유: {reason})" # 사극 말투 적용
            else:
                answer_text = "송구하오나, 답변을 마련하는 중에 문제가 생겼사옵니다." # 사극 말투 적용

        return jsonify({"answer": answer_text})

    except Exception as e:
        app.logger.error(f"Gemini API 호출 중 오류: {e}", exc_info=True)
        return jsonify({"error": f"API와 연결하는 중에 문제가 발생하였사옵니다."}), 500 # 사극 말투 적용


@app.route('/guestbook', methods=['GET'])
def get_guestbook():
    entries = load_guestbook_entries()
    entries.reverse()
    return jsonify(entries)


@app.route('/guestbook', methods=['POST'])
def add_guestbook_entry():
    data = request.get_json()
    name = data.get('name', '').strip()
    message = data.get('message', '').strip()

    if not name or not message:
        return jsonify({"error": "성함과 남기실 말씀을 모두 적어주시옵소서."}), 400 # 사극 말투 적용
    if len(name) > 20 or len(message) > 500:
        return jsonify({"error": "성함은 스무 자, 말씀은 오백 자를 넘지 않도록 해주시옵소서."}), 400 # 사극 말투 적용

    timestamp = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')

    new_entry = {
        "name": name,
        "message": message,
        "timestamp": timestamp
    }

    entries = load_guestbook_entries()
    entries.append(new_entry)

    if save_guestbook_entries(entries):
        return jsonify({"success": True, "entry": new_entry}), 201
    else:
        return jsonify({"error": "방명록을 저장하는 중에 문제가 발생하였사옵니다."}), 500 # 사극 말투 적용


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
