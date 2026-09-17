import streamlit as st
import os
import subprocess

# 페이지 기본 설정
st.set_page_config(
    page_title="Equity Lens - AI Multi-Agent Research",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Equity Lens: AI Multi-Agent Stock Research")
st.markdown("토스 증권 API와 다중 AI 에이전트(포트폴리오, 거시경제, 리서치 종합)를 활용한 지능형 주식 분석 시스템입니다.")

# 사이드바 설정 (실행 모드 선택)
st.sidebar.header("⚙️ Execution Settings")
run_mode = st.sidebar.selectbox("모드 선택", ["demo", "live"])

if st.sidebar.button("🚀 에이전트 분석 파이프라인 실행"):
    with st.spinner("AI 에이전트들이 데이터를 수집하고 분석 중입니다... 잠시만 기다려주세요!"):
        try:
            # 기존 app.py를 subprocess로 실행
            result = subprocess.run(
                ["python", "app.py", run_mode],
                capture_output=True,
                text=True,
                check=True
            )
            st.sidebar.success("분석 완료!")
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"실행 중 오류 발생: {e.stderr}")

st.markdown("---")

# 결과물 리포트(마크다운) 불러오기 및 화면 출력
st.subheader("📊 최신 AI 리서치 리포트")

report_path = "data/latest_portfolio.md"

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    # 웹 화면에 마크다운 형식으로 예쁘게 렌더링
    st.markdown(report_content)
else:
    st.info("아직 생성된 리포트가 없습니다. 좌측 사이드바에서 '에이전트 분석 파이프라인 실행' 버튼을 눌러주세요!")