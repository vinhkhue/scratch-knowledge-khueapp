import streamlit as st
import time
from streamlit_agraph import agraph, Node, Edge, Config
import config
from src.query_engine import GraphQueryEngine

# --- Configuration ---
st.set_page_config(
    page_title="Scratch Knowledge Graph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Engine
@st.cache_resource
def get_engine():
    return GraphQueryEngine()

engine = get_engine()

# --- Helper Functions ---
def display_header():
    """Display the main header"""
    st.title("🤖 Scratch Knowledge Graph Assistant")
    st.markdown("""
    Hệ thống hỏi đáp thông minh về lập trình Scratch, sử dụng **Knowledge Graph (Neo4j)** kết hợp với **LLM (GPT-4o)** 
    để cung cấp câu trả lời chính xác, kèm theo ngữ cảnh đồ thị trực quan.
    """)
    st.markdown("---")

def display_sidebar():
    """Display sidebar with options"""
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Display connection status
        st.success("✅ Đã kết nối Neo4j")
        st.success("✅ Đã kết nối OpenAI")
        
        st.markdown("---")
        st.header("ℹ️ Thông tin")
        st.info("""
        **Phiên bản:** 2.1 (Rebuild)
        **Backend:** Neo4j + LangChain Concept
        **Model:** GPT-4o
        """)
        
        st.markdown("---")
        if st.button("🧹 Xóa & Tải lại Dữ liệu (Admin)"):
             st.warning("Vui lòng chạy lệnh terminal: `python src/ingestion.py --wipe`")

        return {}

def render_graph(graph_data):
    """Render the interactive graph using streamlit-agraph"""
    if not graph_data or not graph_data["nodes"]:
        return

    st.subheader("📊 Graph Minh họa")
    
    nodes = [
        Node(
            id=n["id"], 
            label=n["label"], 
            size=25, 
            title=n.get("title", ""), 
            group=n.get("group", "Entity")
        ) for n in graph_data["nodes"]
    ]
    
    edges = [
        Edge(
            source=e["source"], 
            target=e["target"], 
            label=e.get("label", "RELATED"),
            type="CURVE_SMOOTH"
        ) for e in graph_data["edges"]
    ]
    
    config_graph = Config(
        width=700, 
        height=500, 
        directed=True, 
        nodeHighlightBehavior=True, 
        highlightColor="#F7A7A6", 
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': True}
    )
    
    # Wrap in a container for better layout
    with st.container():
        agraph(nodes=nodes, edges=edges, config=config_graph)

def display_results(query, response_text, graph_data, source):
    """Display the search results in a structured format"""
    
    # 1. Main Response
    st.subheader("💬 Câu trả lời")
    
    # Source Indicator
    if source == "Web Search":
        st.info("🌐 **Nguồn: Tìm kiếm Web (Web Search)** - Do không tìm thấy thông tin trong Knowledge Graph.")
    elif source == "AI Knowledge":
        st.warning("🤖 **Nguồn: Mô hình AI (GPT-4o)** - Không tìm thấy thông tin trong Knowledge Graph hoặc Web. Đây là kiến thức tổng quát.")
    elif source == "GraphRAG":
        st.success("✅ **Nguồn: GraphRAG Knowledge Graph**")
        
    if "Xin lỗi" in response_text and not graph_data["nodes"] and source == "GraphRAG":
         st.warning(response_text)
    else:
         st.markdown(response_text)

    # Debug/Info for User
    if source == "GraphRAG":
        st.caption(f"ℹ️ Tìm thấy {len(graph_data['nodes'])} thực thể liên quan trong Knowledge Graph.")
        

    # 2. Graph Visualization
    if graph_data["nodes"]:
        render_graph(graph_data)
        
        # 3. Entity Details Expander
        with st.expander("📋 Chi tiết các thực thể (Nodes)"):
            for n in graph_data["nodes"]:
                st.markdown(f"**{n['label']}** ({n.get('group', 'Entity')})")
                st.caption(n.get("title", "Không có mô tả"))
                st.markdown("---")

def set_query(q):
    """Callback to set query and trigger search"""
    st.session_state.main_query_input = q
    st.session_state.trigger_search = True

def main():
    display_header()
    display_sidebar()

    # Query Input Section
    st.subheader("❓ Đặt câu hỏi")
    
    # Example Questions
    with st.expander("💡 Câu hỏi mẫu"):
        img_cols = st.columns(3)
        example_questions = [
            "Scratch là gì?",
            "Khối lệnh trong Scratch có những loại nào?",
            "Sprite hoạt động như thế nào?",
            "Cách tạo vòng lặp trong Scratch?",
            "Phiên bản mới nhất của Scratch tới ngày hôm nay là gì?"
        ]
        
        # Use columns for buttons nicely
        cols = st.columns(2)
        for i, q in enumerate(example_questions):
            with cols[i % 2]:
                st.button(f"👉 {q}", key=f"ex_{i}", on_click=set_query, args=(q,))

    # Input Box
    query = st.text_input(
        "Nhập câu hỏi của bạn:", 
        value=st.session_state.get("query_input", ""),
        placeholder="Ví dụ: Scratch là ngôn ngữ gì?",
        key="main_query_input"
    )

    # Check trigger from callback or manual button click
    trigger = st.session_state.get("trigger_search", False)
    force_web = st.session_state.get("force_web_search", False)

    if st.button("🔍 Tìm kiếm", type="primary"):
        trigger = True

    if trigger or force_web or (query and query != st.session_state.get("last_query_executed")):
        if query.strip():
            # Reset triggers
            st.session_state.trigger_search = False
            st.session_state.force_web_search = False
            st.session_state.last_query_executed = query
            
            with st.spinner("Đang phân tích câu hỏi & Truy vấn..."):
                # Simulate "thinking" steps for UX
                time.sleep(0.5) 
                
                # Execute Search
                if force_web:
                    # Force the engine to use Web Tool
                    response_text, graph_data, source = engine.search(query, force_web_search=True)
                else:
                    response_text, graph_data, source = engine.search(query)
            
            # Display Results
            display_results(query, response_text, graph_data, source)
        elif trigger:
             st.warning("Vui lòng nhập nội dung câu hỏi!")

if __name__ == "__main__":
    main()
