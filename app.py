import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 網頁基本設定
st.set_page_config(layout="wide", page_title="最小平方法 (Least Squares) 互動教學網站")

st.title("🎓 最小平方法 (Method of Least Squares) 互動式多功能學習平台")
st.markdown("本網站旨在透過**直觀概念、實際應用、動態模擬與習題演練**，帶你完整掌握最小平方法與正規方程。")

# --- 使用分頁 (Tabs) 進行多頁面架構設計 ---
tab1, tab2, tab3, tab4 = st.tabs(["📖 1. 什麼是最小平方法", "🚀 2. 實際應用場景", "📊 3. 核心數據模擬 (矩陣運算)", "📝 4. 隨堂習題演練"])

# ==========================================
# 分頁 1：觀念介紹
# ==========================================
with tab1:
    st.markdown("""
        <style>
            .big-center-text { text-align: center; font-size: 22px !important; line-height: 1.8; }
            .title-center-text { text-align: center; font-size: 32px !important; font-weight: bold; margin-top: 25px; margin-bottom: 20px; color: #FF4B4B; }
            .subtitle-center-text { text-align: center; font-size: 24px !important; font-weight: bold; margin-top: 30px; margin-bottom: 15px; color: #00CC96; }
            .extension-title { text-align: center; font-size: 24px !important; font-weight: bold; margin-top: 35px; margin-bottom: 15px; color: #FF66B2; }
            .purple-highlight { text-align: center; font-size: 20px !important; font-weight: bold; color: #AB47BC; margin-top: 15px; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title-center-text">🤔 核心問題與觀念</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-center-text">在科學實驗、經濟預測或機器學習中，我們常常會收集到一堆觀測數據點。<br>這些點因為測量誤差或隨機干擾，往往不會乖乖地排成一條完美的直線。<br><b>最小平方法（Method of Least Squares）</b> 的目標，就是找到一條「最能代表這群數據趨勢」的直線或最佳擬合曲線。</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="title-center-text">📐 為什麼叫「最小平方」？</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-center-text">1. <b>殘差 (Residual)</b>：每個真實數據點的 y 值，跟我們預測線上的 y 值，中間的「垂直距離}」就叫做殘差。<br>2. <b>消去正負號</b>：誤差有正有負，直接相加會互相抵消。為了解決這個問題，我們把每個誤差都拿去<b>平方</b>。<br>3. <b>目標</b>：找出一個參數，使得<b>所有點的誤差平方和（SSR）達到最小</b>。</div>', unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # ─── 數學核心正規方程圖解 ───
    st.markdown('<div class="title-center-text">💡 數學核心：正規方程 (Normal Equation) 與矩陣結構</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-center-text">當我們有 N 筆數據點想要擬合最基礎的直線 y = ax + b 時：</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subtitle-center-text">1. 將數據打包成目標向量 y 與設計矩陣 X</div>', unsafe_allow_html=True)
    st.latex(r"y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}, \quad X = \begin{pmatrix} x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_n & 1 \end{pmatrix}")
    
    st.markdown('<div class="subtitle-center-text">2. 待求參數向量 &beta; （斜率 a 在上，截距 b 在下）</div>', unsafe_allow_html=True)
    st.latex(r"\beta = \begin{pmatrix} a \\ b \end{pmatrix}")
    
    st.markdown('<div class="subtitle-center-text">3. 透過著名的正規方程，一瞬間直接解出答案</div>', unsafe_allow_html=True)
    st.latex(r"\beta = (X^T X)^{-1} X^T y")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # ─── n 次多項式函數曲線擴充 ───
    st.markdown('<div class="extension-title">🚀 進階擴充：如果是 n 次多項式函數 (Polynomial Function) 曲線怎麼辦？</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-center-text">如果數據分布呈現非常複雜的波浪、彎曲形狀，直線模型顯然不夠用。<br>此時，我們可以將模型直接擴充為更通用的 <b>n 次多項式曲線模型</b>：</div>', unsafe_allow_html=True)
    st.latex(r"y = c_n x^n + c_{n-1} x^{n-1} + \dots + c_1 x + c_0")
    
    st.markdown('<div class="big-center-text">雖然圖形是高度彎曲的曲線，但只要注意到我們要尋找的未知參數間，<b>依然是線性組合！</b><br>也就是說，我們<b>完全不需要修改正規方程公式</b>，只需將設計矩陣 X 的寬度往左橫向擴展即可：</div>', unsafe_allow_html=True)
    st.latex(r"X = \begin{pmatrix} x_1^n & x_1^{n-1} & \dots & x_1 & 1 \\ x_2^n & x_2^{n-1} & \dots & x_2 & 1 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ x_m^n & x_m^{n-1} & \dots & x_m & 1 \end{pmatrix}, \quad \beta = \begin{pmatrix} c_n \\ c_{n-1} \\ \vdots \\ c_1 \\ c_0 \end{pmatrix}")
    
    st.markdown('<div class="purple-highlight">不論你想要擬合二次方、三次方、還是高達 n 次方的曲線，不需更換演算法，<br>一律直接套用「最小平方法」就能瞬間解出多項式的所有係數。<br>這正是為什麼最小平方法能成為機器學習、數據統計中的強大原因。</div>', unsafe_allow_html=True)


# ==========================================
# 分頁 2：應用場景 (精準修正文字顏色 🎨)
# ==========================================
with tab2:
    st.header("🚀 最小平方法可以應用在哪裡？")
    
    # 🌟 用 HTML 將該行文字強制指定為純黑色 (color: #000000)，取代原本灰色的 st.caption
    st.markdown('<p style="color: #000000; font-size: 16px; margin-bottom: 15px;">最小平方法是現代機器學習（線性回歸）與數據分析的基石，以下是幾個經典的跨領域應用：</p>', unsafe_allow_html=True)
    
    col2_1, col2_2, col2_3 = st.columns(3)
    with col2_1:
        st.subheader("🏠 1. 經濟與房價預測")
        st.markdown("""
        * **情境**：收集某社區過去 50 筆房屋的「坪數 $(X)$」與「成交總價 $(Y)$」。
        * **應用**：利用最小平方法找出一條 $Y = mX + c$ 的直線。未來輸入新房屋的坪數，就能預測出合理的市場價格。
        """)
    with col2_2:
        st.subheader("📈 2. 財務金融 (Beta 係數)")
        st.markdown("""
        * **情境**：比較「某支股票的每日報酬率 $(Y)$」與「大盤市場的每日報酬率 $(X)$」。
        * **應用**：利用最小平方法擬合出的直線斜率，可以用來衡量這支股票相對於市場的波動風險。
        """)
    with col2_3:
        st.subheader("🧪 3. 科學實驗與物理定律")
        st.markdown("""
        * **情境**：物理學家測量彈簧的「受力 $(X)$」與「伸長量 $(Y)$」，但因為儀器老舊有些許誤差。
        * **應用**：將數據透過最小平方法擬合，其直線斜率就是該彈簧的準確**彈性係數**（虎克定律中的 $k$）。
        """)


# ==========================================
# 分頁 3：動態數據模擬
# ==========================================
with tab3:
    st.header("📊 核心技術展示：幾何與誤差地形聯動模擬器")
    
    st.markdown("#### 🛠️ 第一步：設定環境數據（產生隨機數據點）")
    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    n_points = c_col1.slider("數據點數量 (N)", 5, 100, 20, 1)
    noise = c_col2.slider("隨機雜訊強度", 0.0, 10.0, 2.0, 0.1)
    true_m = c_col3.slider("原始真實斜率", -5.0, 5.0, 1.5, 0.5)
    true_c = c_col4.slider("原始真實截距", -10.0, 10.0, 2.0, 0.5)
    
    # --- 後台核心數據與矩陣運算 ---
    np.random.seed(42)
    X_raw = np.linspace(-3, 3, n_points)
    y_raw = true_m * X_raw + true_c + np.random.normal(0, noise, size=n_points)
    
    X_matrix = np.vstack([X_raw, np.ones(len(X_raw))]).T
    Y_vector = y_raw.reshape(-1, 1)
    X_transpose = X_matrix.T
    XT_X = np.dot(X_transpose, X_matrix)
    XT_X_inv = np.linalg.inv(XT_X)
    XT_Y = np.dot(X_transpose, Y_vector)
    Beta_calc = np.dot(XT_X_inv, XT_Y)
    
    calc_slope = Beta_calc[0][0]      
    calc_intercept = Beta_calc[1][0]  
    ssr_best = np.sum((y_raw - (calc_slope * X_raw + calc_intercept))**2)

    st.markdown("---")
    st.markdown("#### 🕹️ 第二步：動態互動探索（可以拉動滑桿，或點擊下方按鈕進行 0.01 微調）")
    
    # 建立精確限制邊界
    min_a, max_a = float(calc_slope - 4.0), float(calc_slope + 4.0)
    min_b, max_b = float(calc_intercept - 5.0), float(calc_intercept + 5.0)

    # 確保滑桿的初始狀態在 key 中建立
    if "slider_a" not in st.session_state:
        st.session_state.slider_a = round(float(calc_slope + 2.5), 2)
    if "slider_b" not in st.session_state:
        st.session_state.slider_b = round(float(calc_intercept - 3.5), 2)

    # 定義加減按鈕的 Callback 函式，直接精準控制 slider 狀態
    def adjust_a(delta):
        st.session_state.slider_a = round(max(min_a, min(max_a, st.session_state.slider_a + delta)), 2)

    def adjust_b(delta):
        st.session_state.slider_b = round(max(min_b, min(max_b, st.session_state.slider_b + delta)), 2)

    guess_col1, guess_col2 = st.columns(2)
    
    with guess_col1:
        current_a = st.slider(
            "🙋‍♂️ 手動調整你的直線斜率 (a)", 
            min_value=min_a, max_value=max_a, 
            key="slider_a",
            step=0.01
        )
        btn_col1, btn_col2 = st.columns(2)
        btn_col1.button("➖ 斜率降 0.01", on_click=adjust_a, args=(-0.01,), use_container_width=True)
        btn_col2.button("➕ 斜率升 0.01", on_click=adjust_a, args=(0.01,), use_container_width=True)
            
    with guess_col2:
        current_b = st.slider(
            "🙋‍♂️ 手動調整你的直線截距 (b)", 
            min_value=min_b, max_value=max_b, 
            key="slider_b",
            step=0.01
        )
        btn_col3, btn_col4 = st.columns(2)
        btn_col3.button("➖ 截距降 0.01", on_click=adjust_b, args=(-0.01,), use_container_width=True)
        btn_col4.button("➕ 截距升 0.01", on_click=adjust_b, args=(0.01,), use_container_width=True)

    # 計算預測線與誤差
    y_user_fitted = current_a * X_raw + current_b
    ssr_user = np.sum((y_raw - y_user_fitted) ** 2)

    st.markdown("---")

    # 左右並排區塊
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.subheader("📈 2D 數據最佳回歸線")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_raw, y=y_raw, mode='markers', name='原始數據點', marker=dict(color='#00CC96', size=10)))
        
        x_line = np.linspace(-4, 4, 100)
        y_best_line = calc_slope * x_line + calc_intercept
        fig.add_trace(go.Scatter(x=x_line, y=y_best_line, mode='lines', name='📊 數學最佳回歸線', line=dict(color='#EF553B', width=2, dash='dash')))
        
        y_user_line = current_a * x_line + current_b
        fig.add_trace(go.Scatter(x=x_line, y=y_user_line, mode='lines', name='🕹️ 你目前控制的直線', line=dict(color='#17BECF', width=4)))
        
        for i in range(len(X_raw)):
            fig.add_trace(go.Scatter(x=[X_raw[i], X_raw[i]], y=[y_raw[i], y_user_fitted[i]], mode='lines', line=dict(color='rgba(171, 71, 188, 0.6)', width=1.5), showlegend=False))
        
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0, y=1))
        st.plotly_chart(fig, use_container_width=True)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(label="你的當前總誤差 (SSR)", value=f"{ssr_user:.2f}", delta=f"{ssr_user - ssr_best:.2f}", delta_color="inverse")
        metric_col2.metric(label="你的斜率 a / 最佳斜率", value=f"{current_a:.2f}", delta=f"最佳: {calc_slope:.2f}")
        metric_col3.metric(label="你的截距 b / 最佳截距", value=f"{current_b:.2f}", delta=f"最佳: {calc_intercept:.2f}")

    with graph_col2:
        st.subheader("🗺️ 3D 誤差平方和 (SSR) 谷底地形圖")
        
        # 網格範圍
        m_mesh = np.linspace(-10.0, 10.0, 40)
        c_mesh = np.linspace(-20.0, 20.0, 40)
        M, C = np.meshgrid(m_mesh, c_mesh)
        
        Z_ssr = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                Z_ssr[i, j] = np.sum((y_raw - (M[i, j] * X_raw + C[i, j])) ** 2)

        max_z_limit = float(np.max(Z_ssr))

        fig_3d = go.Figure()
        fig_3d.add_trace(go.Surface(x=m_mesh, y=c_mesh, z=Z_ssr, colorscale='Viridis', showscale=True, name='SSR 誤差表面', opacity=0.7))
        
        # 🌟 【優化 1】將數學最佳解（十字）放大並更改為超對比螢光青色，微調 Z 軸使其稍微懸空，確保絕對不被遮擋
        fig_3d.add_trace(go.Scatter3d(
            x=[calc_slope], y=[calc_intercept], z=[ssr_best + (max_z_limit * 0.005)], 
            mode='markers', name='🎯 數學最佳解 (谷底)',
            marker=dict(
                size=14,                  # 大幅放大
                color='#00FFFF',          # 換成超亮螢光藍 (Cyan)
                symbol='cross', 
                line=dict(color='black', width=3)
            )
        ))
        
        # 🌟 【優化 2】將操作者當前位置（圓球）設定為半透明，讓藏在裡面的十字可以直接透視穿出來！
        fig_3d.add_trace(go.Scatter3d(
            x=[current_a], y=[current_b], z=[ssr_user],
            mode='markers', name='🙋‍♂️ 你目前的位置',
            marker=dict(
                size=12, 
                color='#FF4B4B', 
                opacity=0.6,             # 開啟半透明效果 (透明度 60%)
                symbol='circle', 
                line=dict(color='#FFFFFF', width=2.5) # 加強白色外框線
            )
        ))

        # 視角防翻轉與鎖定核心設定
        fig_3d.update_layout(
            template="plotly_dark", height=450,
            scene=dict(
                xaxis=dict(title="斜率 a", range=[-10.0, 10.0], autorange=False),
                yaxis=dict(title="截距 b", range=[-20.0, 20.0], autorange=False),
                zaxis=dict(title="SSR 誤差值", range=[0, max_z_limit], autorange=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),       
                    center=dict(x=0, y=0, z=-0.1) 
                )
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0, y=1),
            uirevision='permanently_locked'         
        )
        
        st.plotly_chart(fig_3d, use_container_width=True, key="permanently_stable_3d_chart")

    # 底下矩陣算式
    st.markdown("---")
    st.markdown("### 📝 線性代數運算詳細變換步驟 (Matric Calculation Logs)")
    calc_col1, calc_col2 = st.columns(2)
    with calc_col1:
        st.markdown("**1. 建立設計矩陣 $X$ 與目標向量 $\\mathbf{y}$（展示前3筆）**")
        st.latex(rf"X = \begin{{pmatrix}} {X_matrix[0,0]:.2f} & 1 \\ {X_matrix[1,0]:.2f} & 1 \\ {X_matrix[2,0]:.2f} & 1 \\ \vdots & \vdots \end{{pmatrix}}, \quad \mathbf{{y}} = \begin{{pmatrix}} {Y_vector[0,0]:.2f} \\ {Y_vector[1,0]:.2f} \\ {Y_vector[2,0]:.2f} \\ \vdots \end{{pmatrix}}")
        st.markdown("**2. 計算轉置矩陣相乘：共變異矩陣 $X^T X$**")
        st.latex(rf"X^T X = \begin{{pmatrix}} {XT_X[0,0]:.1f} & {XT_X[0,1]:.1f} \\ {XT_X[1,0]:.1f} & {XT_X[1,1]:.1f} \end{{pmatrix}}")
    with calc_col2:
        st.markdown("**3. 執行逆矩陣運算 $(X^T X)^{-1}$**")
        st.latex(rf"(X^T X)^{{-1}} = \begin{{pmatrix}} {XT_X_inv[0,0]:.4f} & {XT_X_inv[0,1]:.4f} \\ {XT_X_inv[0,1]:.4f} & {XT_X_inv[1,1]:.4f} \end{{pmatrix}}")
        st.markdown("**4. 求解參數向量 $\\beta$**")
        st.latex(rf"\beta = (X^T X)^{{-1}} X^T \mathbf{{y}} = \begin{{pmatrix}} {calc_slope:.4f} \\ {calc_intercept:.4f} \end{{pmatrix}}")
        st.success(f"🎉 最佳參數解： 斜率 a = {calc_slope:.2f} ｜ 截距 b = {calc_intercept:.2f}")

# ==========================================
# 分頁 4：習題演練
# ==========================================
with tab4:
    if "q1_correct" not in st.session_state: st.session_state.q1_correct = False
    if "q2_correct" not in st.session_state: st.session_state.q2_correct = False
    st.header("📝 最小平方法隨堂挑戰")
    st.markdown("---")
    q1 = st.radio("**Q1. 最小平方法中所要最小化的「平方和」，指的是什麼的平方和？**", ["A) 原始數據 X 值的平方和", "B) 原始數據 Y 值的平方和", "C) 預測值與真實值之間的「殘差（誤差）」平方和", "D) 斜率與截距的平方和"])
    if st.button("檢查 Q1 答案"):
        if q1 == "C) 預測值與真實值之間的「殘差（誤差）」平方和":
            st.session_state.q1_correct = True
            st.success("🎯 Q1 正確！")
        else:
            st.error("❌ 答錯囉，再想想看！")
    elif st.session_state.q1_correct: st.success("🎯 Q1 正確！")

    st.markdown("---")
    st.markdown("**Q2. 講義真實期末考題實測 📝**")
    st.markdown("已知有五筆觀測數據為：**(1, 1), (2, 3), (3, 4), (4, 6), (5, 5)**。請計算出最符合這五筆資料的最小平方回歸線 $y = ax + b$。")
    ans_a = st.number_input("請輸入你算出來的斜率 (a)：", value=0.0, step=0.1)
    ans_b = st.number_input("請輸入你算出來的截距 (b)：", value=0.0, step=0.1)
    if st.button("提交講義題目答案"):
        if abs(ans_a - 1.1) < 0.01 and abs(ans_b - 0.5) < 0.01:
            st.session_state.q2_correct = True
            st.success("🎉 Q2 正確！")
        else:
            st.error("❌ 數字不對喔，再檢查一下計算。")
    elif st.session_state.q2_correct: st.success("🎉 Q2 正確！")

    if st.session_state.q1_correct and st.session_state.q2_correct:
        st.markdown("---")
        st.info("🏆 **恭喜你已經完全掌握了最小平方法的所有核心觀念與計算！** 👏👏👏")