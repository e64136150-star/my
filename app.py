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
# 分頁 1：觀念介紹 (n 次多項式通用版本)
# ==========================================
with tab1:
    # 透過 CSS 強力將分頁 1 的字體與公式全部放大並置中
    st.markdown("""
        <style>
            .big-center-text {
                text-align: center;
                font-size: 22px !important;
                line-height: 1.8;
            }
            .title-center-text {
                text-align: center;
                font-size: 32px !important;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 20px;
                color: #FF4B4B;
            }
            .subtitle-center-text {
                text-align: center;
                font-size: 26px !important;
                font-weight: bold;
                margin-top: 30px;
                margin-bottom: 15px;
                color: #00CC96;
            }
            /* 讓 LaTeX 公式也跟著置中與放大 */
            .katex-display {
                font-size: 26px !important;
                text-align: center !important;
                margin: 20px 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 1. 核心問題與定義 (置中放大)
    st.markdown('<div class="title-center-text">🤔 核心問題與觀念</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="big-center-text">
    在科學實驗、經濟預測或機器學習中，我們常常會收集到一堆觀測數據點。<br>
    這些點因為測量誤差或隨機干擾，往往不會乖乖地排成一條完美的直線。<br>
    <b>最小平方法（Method of Least Squares）</b> 的目標，就是找到一條「最能代表這群數據趨勢」的直線或最佳擬合曲線。
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="title-center-text">📐 為什麼叫「最小平方」？</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="big-center-text">
    1. <b>殘差 (Residual)</b>：每個真實數據點的 y 值，跟我們預測線上的 y 值，中間的「垂直距離」就叫做殘差。<br>
    2. <b>消去正負號</b>：誤差有正有負，直接相加會互相抵消。為了解決這個問題，我們把每個誤差都拿去<b>平方</b>。<br>
    3. <b>目標</b>：找出一個參數，使得<b>所有點的誤差平方和（SSR）達到最小</b>。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 2. 數學核心
    st.markdown('<div class="title-center-text">💡 數學核心：正規方程 (Normal Equation) 與矩陣結構</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-center-text">當我們有 N 筆數據點想要擬合最基礎的直線 <b>y = ax + b</b> 時：</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subtitle-center-text">1. 將數據打包成目標向量 y 與設計矩陣 X</div>', unsafe_allow_html=True)
    st.latex(r"\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}, \quad X = \begin{pmatrix} x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_n & 1 \end{pmatrix}")
    
    st.markdown('<div class="subtitle-center-text">2. 待求參數向量 β（斜率 a 在上，截距 b 在下）</div>', unsafe_allow_html=True)
    st.latex(r"\boldsymbol{\beta} = \begin{pmatrix} a \\ b \end{pmatrix}")
    
    st.markdown('<div class="subtitle-center-text">3. 透過著名的正規方程，一瞬間直接解出答案</div>', unsafe_allow_html=True)
    st.latex(r"\boldsymbol{\beta} = (X^T X)^{-1} X^T \mathbf{y}")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # 3. n 次多項式延伸問題
    st.markdown('<div class="title-center-text">🚀 進階擴充：如果是 n 次多項式函數（Polynomial Function）曲線怎麼辦？</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="big-center-text">
    如果數據分布呈現非常複雜的波浪、彎曲形狀，直線模型顯然不夠用。<br>
    此時，我們可以將模型直接擴充為更通用的 <b>n 次多項式曲線模型</b>：
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"y = c_n x^n + c_{n-1} x^{n-1} + \dots + c_1 x + c_0")
    
    st.markdown("""
    <div class="big-center-text">
    雖然圖形是高度彎曲的曲線，但只要注意到我們要尋找的未知參數間，依然是<b>線性組合</b>！<br>
    也就是說，我們<b>完全不需要修改正規方程公式</b>，只需將<b>設計矩陣 X</b> 的寬度往左橫向擴展即可：
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"""
    X = \begin{pmatrix} 
    x_1^n & x_1^{n-1} & \dots & x_1 & 1 \\ 
    x_2^n & x_2^{n-1} & \dots & x_2 & 1 \\ 
    \vdots & \vdots & \ddots & \vdots & \vdots \\ 
    x_m^n & x_m^{n-1} & \dots & x_m & 1 
    \end{pmatrix}, \quad 
    \boldsymbol{\beta} = \begin{pmatrix} c_n \\ c_{n-1} \\ \vdots \\ c_1 \\ c_0 \end{pmatrix}
    """)
    
    st.markdown("""
    <div class="big-center-text" style="color: #AB47BC; font-weight: bold; margin-bottom: 50px;">
    不論你想要擬合二次方、三次方、還是高達 n 次方的曲線，不需更換演算法，<br>
    一律直接套用「最小平方法」就能瞬間解出多項式的所有係數。<br>
    這正是為什麼最小平方法能成為機器學習、數據統計中的強大原因。
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 分頁 2：應用場景
# ==========================================
with tab2:
    st.header("🚀 最小平方法可以應用在哪裡？")
    st.write("最小平方法是現代機器學習（線性回歸）與數據分析的基石，以下是幾個經典的跨領域應用：")
    
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
        * **應用**：將數據透過最小平方法擬合，其直線斜率就是該彈簧的準確**彈性係數（虎克定律中的 $k$）**。
        """)

# ==========================================
# 分頁 3：動態數據模擬
# ==========================================
with tab3:
    st.header("📊 核心技術展示：正規方程矩陣模擬器")
    st.caption("請在左側控制面板調整參數，右側的線性代數矩陣運算與左側圖表會即時同步變更。")
    
    sim_col1, sim_col2 = st.columns([3, 2])
    
    with sim_col1:
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        
        n_points = c_col1.slider("數據點數量 (N)", 5, 100, 20, 1)
        noise = c_col2.slider("隨機雜訊強度", 0.0, 10.0, 2.0, 0.1)
        true_m = c_col3.slider("真實斜率 (m)", -5.0, 5.0, 1.5, 0.5)
        true_c = c_col4.slider("真實截距 (c)", -10.0, 10.0, 2.0, 0.5)
        
        np.random.seed(42)
        X_raw = np.linspace(-5, 5, n_points)
        y_raw = true_m * X_raw + true_c + np.random.normal(0, noise, size=n_points)
        
        # --- 核心矩陣運算 ---
        X_matrix = np.vstack([X_raw, np.ones(len(X_raw))]).T
        Y_vector = y_raw.reshape(-1, 1)
        X_transpose = X_matrix.T
        XT_X = np.dot(X_transpose, X_matrix)
        
        try:
            XT_X_inv = np.linalg.inv(XT_X)
            is_invertible = True
        except np.linalg.LinAlgError:
            is_invertible = False
            
        if is_invertible:
            XT_Y = np.dot(X_transpose, Y_vector)
            Beta = np.dot(XT_X_inv, XT_Y)
            calc_slope = Beta[0][0]      
            calc_intercept = Beta[1][0]  
            y_fitted = np.dot(X_matrix, Beta).flatten()
            residuals = y_raw - y_fitted
            ssr = np.sum(residuals**2)

        # 繪圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_raw, y=y_raw, mode='markers', name='原始數據點 (y)', marker=dict(color='#00CC96', size=10)))
        if is_invertible:
            x_line = np.linspace(-6, 6, 100)
            y_line = calc_slope * x_line + calc_intercept
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='最小平方法回歸線', line=dict(color='#EF553B', width=3)))
            for i in range(len(X_raw)):
                fig.add_trace(go.Scatter(x=[X_raw[i], X_raw[i]], y=[y_raw[i], y_fitted[i]], mode='lines', line=dict(color='rgba(171, 71, 188, 0.5)', width=1), showlegend=False))
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # 即時數據提取面板
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric(label="當前殘差平方和 (SSR)", value=f"{ssr:.4f}")
        metric_col2.metric(label="📊 擬合最佳斜率 (a)", value=f"{calc_slope:.2f}")
        metric_col3.metric(label="🧱 擬合最佳截距 (b)", value=f"{calc_intercept:.2f}")

    with sim_col2:
        st.markdown("#### ⚡ 矩陣運算變換步驟")
        
        st.markdown("**1. 建立設計矩陣 $X$ 與目標向量 $\\mathbf{y}$**")
        st.latex(r"X = \begin{pmatrix} x_1 & 1 \\ x_2 & 1 \\ x_3 & 1 \\ \vdots & \vdots \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ y_3 \\ \vdots \end{pmatrix}")
        st.markdown("帶入當前數據之實際矩陣（展示前3筆）：")
        st.latex(rf"""
        X = \begin{{pmatrix}} 
        {X_matrix[0,0]:.2f} & 1 \\ 
        {X_matrix[1,0]:.2f} & 1 \\ 
        {X_matrix[2,0]:.2f} & 1 \\ 
        \vdots & \vdots 
        \end{{pmatrix}}, \quad 
        \mathbf{{y}} = \begin{{pmatrix}} 
        {Y_vector[0,0]:.2f} \\ 
        {Y_vector[1,0]:.2f} \\ 
        {Y_vector[2,0]:.2f} \\ 
        \vdots 
        \end{{pmatrix}}
        """)
        
        st.markdown("**2. 共變異矩陣 $X^T X$**")
        st.latex(rf"X^T X = \begin{{pmatrix}} {XT_X[0,0]:.1f} & {XT_X[0,1]:.1f} \\ {XT_X[1,0]:.1f} & {XT_X[1,1]:.1f} \end{{pmatrix}}")
        
        st.markdown("**3. 執行逆矩陣運算 $(X^T X)^{-1}$**")
        st.latex(rf"(X^T X)^{{-1}} = \begin{{pmatrix}} {XT_X_inv[0,0]:.4f} & {XT_X_inv[0,1]:.4f} \\ {XT_X_inv[1,0]:.4f} & {XT_X_inv[1,1]:.4f} \end{{pmatrix}}")
        
        st.markdown("**4. 計算參數解向量 $\\boldsymbol{\\beta}$ 矩陣結果**")
        st.latex(rf"""
        \boldsymbol{{\beta}} = (X^T X)^{{-1}} X^T \mathbf{{y}} = \begin{{pmatrix}} {calc_slope:.4f} \\ {calc_intercept:.4f} \end{{pmatrix}}
        """)
        
        st.markdown("**5. 最終擬合參數提取**")
        st.success(f"得出最佳解： 斜率 (a) = {calc_slope:.2f} ｜ 截距 (b) = {calc_intercept:.2f}")

# ==========================================
# 分頁 4：習題演練 (新增 Session State 記憶功能)
# ==========================================
with tab4:
    # 初始化答對狀態（如果沒建立過，預設為 False）
    if "q1_correct" not in st.session_state:
        st.session_state.q1_correct = False
    if "q2_correct" not in st.session_state:
        st.session_state.q2_correct = False

    st.header("📝 最小平方法隨堂挑戰")
    st.write("學完觀念與模擬後，來測試看看你的理解程度吧！")
    
    st.markdown("---")
    # 題目一
    q1 = st.radio(
        "**Q1. 最小平方法中所要最小化的「平方和」，指的是什麼的平方和？**",
        ["A) 原始數據 X 值的平方和", "B) 原始數據 Y 值的平方和", "C) 預測值與真實值之間的「殘差（誤差）」平方和", "D) 斜率與截距的平方和"]
    )
    if st.button("檢查 Q1 答案"):
        if q1 == "C) 預測值與真實值之間的「殘差（誤差）」平方和":
            st.session_state.q1_correct = True
            st.success("🎯 Q1 正確！最小平方法的目標就是讓殘差的平方和（SSR）達到最小。")
        else:
            st.session_state.q1_correct = False
            st.error("❌ 答錯囉，再想想看！提示：觀察分頁 3 中連結點與線之間的紫線。")
            
    # 如果先前已經答對，即使沒按按鈕也讓提示留著
    elif st.session_state.q1_correct:
        st.success("🎯 Q1 正確！最小平方法的目標就是讓殘差的平方和（SSR）達到最小。")

    st.markdown("---")
    
    # Q2 隨堂挑戰
    st.markdown("**Q2. 實際例題 📝**")
    st.markdown("已知有五筆觀測數據為：**(1, 1), (2, 3), (3, 4), (4, 6), (5, 5)**。請計算出最符合這五筆資料的最小平方回歸線 $y = ax + b$。")
    
    ans_a = st.number_input("請輸入你算出來的斜率 (a)：", value=0.0, step=0.1)
    ans_b = st.number_input("請輸入你算出來的截距 (b)：", value=0.0, step=0.1)
    
    if st.button("提交講義題目答案"):
        if abs(ans_a - 1.1) < 0.01 and abs(ans_b - 0.5) < 0.01:
            st.session_state.q2_correct = True
            st.success("🎉 Q2 正確！（斜率 a = 1.1, 截距 b = 0.5）")
        else:
            st.session_state.q2_correct = False
            st.error("❌ 數字不對喔，再檢查一下計算。提示：可以試著套用正規方程 $\\boldsymbol{\\beta} = (X^T X)^{-1} X^T \\mathbf{y}$ 解解看！")
            
    # 如果先前已經答對，即使沒按按鈕也讓提示留著
    elif st.session_state.q2_correct:
        st.success("🎉 Q2 正確！（斜率 a = 1.1, 截距 b = 0.5）")

    # ==========================================
    # 核心亮點：當兩題都答對時，永久顯示大獎勵區塊！
    # ==========================================
    if st.session_state.q1_correct and st.session_state.q2_correct:
        st.markdown("---")
        st.info("🏆 **恭喜你已經完全掌握了最小平方法的所有核心觀念與計算！** 👏👏👏")