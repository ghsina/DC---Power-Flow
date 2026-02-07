import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="DC Power Flow Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .matrix-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ DC Power Flow Calculator ⚡</h1>', unsafe_allow_html=True)
st.markdown("### An Interactive Tool for Power System Analysis")
st.markdown("---")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'num_nodes' not in st.session_state:
    st.session_state.num_nodes = 3
if 'reactances' not in st.session_state:
    st.session_state.reactances = {}
if 'powers' not in st.session_state:
    st.session_state.powers = {}
if 'ref_bus' not in st.session_state:
    st.session_state.ref_bus = 1

# Sidebar for navigation
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Power+Flow", use_column_width=True)
    st.markdown("### 📋 Navigation")
    st.markdown(f"**Current Step:** {st.session_state.step}/5")
    
    progress = st.session_state.step / 5
    st.progress(progress)
    
    st.markdown("---")
    st.markdown("### 📖 Steps:")
    st.markdown("1️⃣ Set number of nodes")
    st.markdown("2️⃣ Enter line reactances")
    st.markdown("3️⃣ Enter power injections")
    st.markdown("4️⃣ Select reference bus")
    st.markdown("5️⃣ View results")
    
    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Step 1: Number of Nodes
if st.session_state.step == 1:
    st.markdown('<div class="step-header"><h2>Step 1: Define Your Power System 🔌</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_nodes = st.number_input(
            "Enter the number of nodes (buses) in your system:",
            min_value=2,
            max_value=20,
            value=3,
            step=1,
            help="Typical systems have 3-10 nodes for educational purposes"
        )
        
        st.info(f"📊 You'll need to define **{num_nodes * (num_nodes - 1) // 2}** line reactances for a fully connected {num_nodes}-node system.")
        
        if st.button("Next ➡️", type="primary"):
            st.session_state.num_nodes = num_nodes
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        st.markdown("#### 💡 Quick Guide")
        st.markdown("""
        - **Node/Bus**: Connection point in the grid
        - **Typical systems**: 3-10 nodes
        - **Example**: 3-node system (Generator, Load, Junction)
        """)

# Step 2: Line Reactances
elif st.session_state.step == 2:
    st.markdown('<div class="step-header"><h2>Step 2: Enter Line Reactances (X<sub>ij</sub>) ⚙️</h2></div>', unsafe_allow_html=True)
    
    n = st.session_state.num_nodes
    
    st.markdown(f"### Enter reactance values for lines between nodes (in per unit or Ω)")
    
    # Create a grid layout for inputs
    connections = [(i, j) for i in range(1, n+1) for j in range(i+1, n+1)]
    
    cols_per_row = 3
    for idx in range(0, len(connections), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, (i, j) in enumerate(connections[idx:idx+cols_per_row]):
            with cols[col_idx]:
                key = f"X_{i}_{j}"
                default_val = st.session_state.reactances.get(key, 0.1)
                value = st.number_input(
                    f"X_{i}{j} (Node {i} ↔ Node {j})",
                    min_value=0.001,
                    max_value=10.0,
                    value=default_val,
                    step=0.01,
                    format="%.3f",
                    key=key
                )
                st.session_state.reactances[key] = value
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next ➡️", type="primary"):
            st.session_state.step = 3
            st.rerun()

# Step 3: Power Injections
elif st.session_state.step == 3:
    st.markdown('<div class="step-header"><h2>Step 3: Enter Power Injections (P<sub>i</sub>) 🔋</h2></div>', unsafe_allow_html=True)
    
    n = st.session_state.num_nodes
    
    st.markdown("### Enter net power injection at each node (in MW or per unit)")
    st.info("💡 **Positive** = Generation, **Negative** = Load, **Zero** = Transit node")
    
    cols = st.columns(min(4, n))
    for i in range(1, n+1):
        with cols[(i-1) % len(cols)]:
            key = f"P_{i}"
            default_val = st.session_state.powers.get(key, 0.0)
            value = st.number_input(
                f"P_{i} (Node {i})",
                min_value=-1000.0,
                max_value=1000.0,
                value=default_val,
                step=1.0,
                format="%.2f",
                key=key
            )
            st.session_state.powers[key] = value
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Next ➡️", type="primary"):
            st.session_state.step = 4
            st.rerun()

# Step 4: Reference Bus
elif st.session_state.step == 4:
    st.markdown('<div class="step-header"><h2>Step 4: Select Reference Bus 🎯</h2></div>', unsafe_allow_html=True)
    
    n = st.session_state.num_nodes
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Choose which node will be the reference (slack) bus")
        
        ref_bus = st.selectbox(
            "Reference Bus Number:",
            options=list(range(1, n+1)),
            index=0,
            help="The reference bus voltage angle is set to 0°"
        )
        
        st.session_state.ref_bus = ref_bus
        
        st.success(f"✅ Node {ref_bus} will be the reference bus with θ_{ref_bus} = 0°")
    
    with col2:
        st.markdown("#### 📚 About Reference Bus")
        st.markdown("""
        - Also called **slack bus**
        - Voltage angle = 0° (reference)
        - Usually the main generator
        - Balances system power
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Calculate Results 🚀", type="primary"):
            st.session_state.step = 5
            st.rerun()

# Step 5: Results and Calculations
elif st.session_state.step == 5:
    st.markdown('<div class="step-header"><h2>Step 5: Results & Analysis 📊</h2></div>', unsafe_allow_html=True)
    
    n = st.session_state.num_nodes
    ref_bus = st.session_state.ref_bus
    
    # Build B matrix
    B = np.zeros((n, n))
    
    for key, value in st.session_state.reactances.items():
        parts = key.split('_')
        i, j = int(parts[1]) - 1, int(parts[2]) - 1
        B[i, j] = -1 / value
        B[j, i] = -1 / value
    
    for i in range(n):
        B[i, i] = -np.sum(B[i, :])
    
    # Build P vector
    P = np.array([st.session_state.powers[f"P_{i+1}"] for i in range(n)])
    
    # Display B Matrix
    st.markdown("### 🔢 Susceptance Matrix [B]")
    df_B = pd.DataFrame(B, 
                        columns=[f"θ_{i+1}" for i in range(n)],
                        index=[f"P_{i+1}" for i in range(n)])
    st.dataframe(df_B.style.highlight_max(axis=0).format("{:.4f}"), use_container_width=True)
    
    # Display equation
    st.markdown("### 📐 Matrix Equation")
    st.latex(r"\begin{bmatrix} P_1 \\ P_2 \\ \vdots \\ P_n \end{bmatrix} = \begin{bmatrix} B_{11} & B_{12} & \cdots & B_{1n} \\ B_{21} & B_{22} & \cdots & B_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ B_{n1} & B_{n2} & \cdots & B_{nn} \end{bmatrix} \begin{bmatrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_n \end{bmatrix}")
    
    # Solve for theta
    try:
        # Remove reference bus row and column
        ref_idx = ref_bus - 1
        B_reduced = np.delete(np.delete(B, ref_idx, 0), ref_idx, 1)
        P_reduced = np.delete(P, ref_idx)
        
        # Solve reduced system
        theta_reduced = np.linalg.solve(B_reduced, P_reduced)
        
        # Insert reference bus angle
        theta = np.insert(theta_reduced, ref_idx, 0)
        
        # Display results
        st.markdown("### ⚡ Voltage Angles (θ)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            results_df = pd.DataFrame({
                'Node': [f"Node {i+1}" for i in range(n)],
                'Angle (θ) [rad]': theta,
                'Angle (θ) [degrees]': np.rad2deg(theta),
                'Type': ['Reference' if i+1 == ref_bus else 'PQ Bus' for i in range(n)]
            })
            st.dataframe(results_df.style.highlight_max(subset=['Angle (θ) [rad]']).format({
                'Angle (θ) [rad]': '{:.6f}',
                'Angle (θ) [degrees]': '{:.4f}'
            }), use_container_width=True)
        
        with col2:
            # Angle bar chart
            fig_angles = go.Figure(data=[
                go.Bar(x=[f"Node {i+1}" for i in range(n)], 
                       y=np.rad2deg(theta),
                       marker_color=['red' if i+1 == ref_bus else 'blue' for i in range(n)],
                       text=np.rad2deg(theta),
                       texttemplate='%{text:.2f}°',
                       textposition='outside')
            ])
            fig_angles.update_layout(
                title="Voltage Angles at Each Node",
                xaxis_title="Node",
                yaxis_title="Angle (degrees)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_angles, use_container_width=True)
        
        # Calculate power flows
        st.markdown("### 🔄 Power Flows on Lines")
        
        flows = []
        for key, X_value in st.session_state.reactances.items():
            parts = key.split('_')
            i, j = int(parts[1]) - 1, int(parts[2]) - 1
            P_ij = (theta[i] - theta[j]) / X_value
            P_ji = -P_ij
            flows.append({
                'Line': f"{i+1} → {j+1}",
                'From Node': i+1,
                'To Node': j+1,
                'Reactance (X)': X_value,
                'Power Flow (P_ij)': P_ij,
                'Direction': '→' if P_ij > 0 else '←',
                'Magnitude': abs(P_ij)
            })
        
        flows_df = pd.DataFrame(flows)
        st.dataframe(flows_df.style.format({
            'Reactance (X)': '{:.4f}',
            'Power Flow (P_ij)': '{:.4f}',
            'Magnitude': '{:.4f}'
        }).background_gradient(subset=['Magnitude'], cmap='YlOrRd'), use_container_width=True)
        
        # Network visualization
        st.markdown("### 🌐 Network Topology with Power Flows")
        
        G = nx.Graph()
        for i in range(1, n+1):
            G.add_node(i, power=st.session_state.powers[f"P_{i}"])
        
        edge_labels = {}
        for key, X_value in st.session_state.reactances.items():
            parts = key.split('_')
            i, j = int(parts[1]), int(parts[2])
            P_ij = (theta[i-1] - theta[j-1]) / X_value
            G.add_edge(i, j, weight=abs(P_ij), reactance=X_value)
            edge_labels[(i, j)] = f"{P_ij:.2f} MW"
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_colors = ['red' if i == ref_bus else 'lightblue' for i in G.nodes()]
        node_sizes = [3000 if i == ref_bus else 2000 for i in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, ax=ax, alpha=0.9)
        
        # Draw edges with width based on power flow
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [5 * w / max_weight for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                              edge_color='gray', ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)
        
        # Add power injections as text
        for node, (x, y) in pos.items():
            power = st.session_state.powers[f"P_{node}"]
            ax.text(x, y-0.15, f"P={power:.1f}", ha='center', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title("Power System Network Diagram", fontsize=18, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Generation", f"{sum([p for p in P if p > 0]):.2f} MW")
        with col2:
            st.metric("Total Load", f"{abs(sum([p for p in P if p < 0])):.2f} MW")
        with col3:
            st.metric("Max Angle Difference", f"{(max(theta) - min(theta)) * 180/np.pi:.2f}°")
        with col4:
            st.metric("Max Line Flow", f"{max([abs(f['Power Flow (P_ij)']) for f in flows]):.2f} MW")
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        # Create downloadable CSV
        export_data = f"DC Power Flow Analysis Results\n\n"
        export_data += f"Number of Nodes: {n}\n"
        export_data += f"Reference Bus: {ref_bus}\n\n"
        export_data += "Voltage Angles:\n"
        export_data += results_df.to_csv(index=False)
        export_data += "\n\nPower Flows:\n"
        export_data += flows_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=export_data,
            file_name="dc_powerflow_results.csv",
            mime="text/csv"
        )
        
    except np.linalg.LinAlgError:
        st.error("❌ Error: The system matrix is singular! Check your reactance values.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back to Settings"):
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("🔄 New Calculation"):
            st.session_state.step = 1
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>⚡ DC Power Flow Calculator | Educational Tool for Power Systems Analysis</p>
    <p>Built with Streamlit 🎈 | Python 🐍 | NumPy | NetworkX</p>
</div>
""", unsafe_allow_html=True)
