/**
 * House Price Prediction - Client Side JavaScript
 */

const API_URL = '/api/predict';

// DOM elements
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const predictionPrice = document.getElementById('predictionPrice');
const priceBreakdown = document.getElementById('priceBreakdown');
const explanationText = document.getElementById('explanationText');
const factorsGrid = document.getElementById('factorsGrid');
const knowledgeGraph = document.getElementById('knowledgeGraph');

// Colors
const COLORS = {
    house: '#3498db',
    factor: '#2ecc71',
    price: '#e74c3c',
    edge: '#95a5a6'
};

// Form submission
form.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = {
        location: document.getElementById('location').value,
        area_m2: parseFloat(document.getElementById('area_m2').value) || 0,
        bedrooms: parseInt(document.getElementById('bedrooms').value) || 1,
        bathrooms: parseInt(document.getElementById('bathrooms').value) || 1,
        floors: parseInt(document.getElementById('floors').value) || 1,
        frontage: document.querySelector('input[name="frontage"]:checked').value === 'true'
    };
    
    // Validate
    if (!formData.location) {
        alert('Vui lòng chọn vị trí!');
        return;
    }
    if (formData.area_m2 <= 0) {
        alert('Vui lòng nhập diện tích hợp lệ!');
        return;
    }
    
    // Show loading
    loading.classList.remove('d-none');
    result.classList.add('d-none');
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Đang xử lý...';
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResult(data);
        } else {
            alert('Lỗi: ' + (data.error || 'Không thể dự đoán'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Đã xảy ra lỗi khi kết nối đến server. Vui lòng thử lại.');
    } finally {
        loading.classList.add('d-none');
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-calculator me-2"></i> Dự đoán giá';
    }
});

function displayResult(data) {
    result.classList.remove('d-none');
    
    // Price
    predictionPrice.innerHTML = `
        ${data.prediction.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")} 
        <span class="currency">triệu VND</span>
    `;
    
    // Price breakdown
    const factors = data.factors;
    priceBreakdown.innerHTML = `
        Giá cơ bản: ${factors.base_price.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ",")} triệu 
        × ${factors.total_factor.toFixed(2)} (hệ số tổng hợp)
    `;
    
    // Explanation
    explanationText.textContent = factors.explanation || 'Phân tích hoàn chỉnh';
    
    // Factors
    renderFactors(data);
    
    // Knowledge Graph
    renderKnowledgeGraph(data.knowledge_graph);
    
    // Scroll to result
    result.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderFactors(data) {
    const factors = data.factors;
    
    const items = [
        { label: 'Vị trí', value: factors.location_factor.toFixed(2), 
          impact: getImpactClass(factors.location_factor) },
        { label: 'Diện tích', value: factors.area_factor.toFixed(2),
          impact: getImpactClass(factors.area_factor) },
        { label: 'Phòng ngủ', value: factors.bedroom_factor.toFixed(2),
          impact: getImpactClass(factors.bedroom_factor) },
        { label: 'Phòng tắm', value: factors.bathroom_factor.toFixed(2),
          impact: getImpactClass(factors.bathroom_factor) },
        { label: 'Số tầng', value: factors.floor_factor.toFixed(2),
          impact: getImpactClass(factors.floor_factor) },
        { label: 'Mặt tiền', value: factors.frontage_factor.toFixed(2),
          impact: getImpactClass(factors.frontage_factor) }
    ];
    
    factorsGrid.innerHTML = items.map(item => `
        <div class="factor-item">
            <div class="factor-label">${item.label}</div>
            <div class="factor-value">${item.value}</div>
            <div class="factor-impact ${item.impact}">${item.impact.toUpperCase()}</div>
        </div>
    `).join('');
}

function getImpactClass(value) {
    if (value >= 1.2) return 'high';
    if (value <= 0.8) return 'low';
    return 'medium';
}

function renderKnowledgeGraph(graphData) {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
        knowledgeGraph.innerHTML = '<p class="text-center text-muted">Không có dữ liệu</p>';
        return;
    }
    
    knowledgeGraph.innerHTML = '';
    
    const width = knowledgeGraph.clientWidth || 600;
    const height = 280;
    const margin = { top: 20, right: 40, bottom: 20, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const svg = d3.select(knowledgeGraph)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    const nodes = graphData.nodes.map(n => ({
        id: n.id,
        label: n.label,
        type: n.type,
        properties: n.properties
    }));
    
    const edges = graphData.edges.map(e => ({
        source: e.source,
        target: e.target,
        label: e.label,
        weight: e.weight || 1
    }));
    
    const nodeMap = {};
    nodes.forEach(n => nodeMap[n.id] = n);
    
    const links = edges.map(e => ({
        source: nodeMap[e.source],
        target: nodeMap[e.target],
        label: e.label,
        weight: e.weight
    }));
    
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links)
            .id(d => d.id)
            .distance(d => 100 - d.weight * 20)
            .strength(0.5))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(innerWidth / 2, innerHeight / 2))
        .force('collision', d3.forceCollide().radius(30));
    
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke', COLORS.edge)
        .attr('stroke-width', d => 1 + d.weight * 0.5)
        .attr('stroke-opacity', 0.6);
    
    const linkLabels = svg.append('g')
        .selectAll('text')
        .data(links)
        .enter()
        .append('text')
        .attr('fill', '#6c757d')
        .attr('font-size', '9px')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .text(d => d.label);
    
    const nodeGroup = svg.append('g')
        .selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    nodeGroup.append('circle')
        .attr('r', d => d.type === 'Chính' ? 28 : d.type === 'Kết quả' ? 25 : 20)
        .attr('fill', d => {
            if (d.type === 'Chính') return COLORS.house;
            if (d.type === 'Kết quả') return COLORS.price;
            return COLORS.factor;
        })
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('opacity', 0.9)
        .on('mouseover', function() {
            d3.select(this).transition().duration(200).attr('r', 30);
        })
        .on('mouseout', function(d) {
            const r = d.type === 'Chính' ? 28 : d.type === 'Kết quả' ? 25 : 20;
            d3.select(this).transition().duration(200).attr('r', r);
        });
    
    nodeGroup.append('text')
        .attr('fill', '#fff')
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .text(d => d.label);
    
    nodeGroup.on('mouseover', function(event, d) {
        const props = d.properties;
        const tooltipText = Object.entries(props)
            .map(([k, v]) => `${k}: ${v}`)
            .join('\n');
        const node = d3.select(this);
        node.append('title').text(tooltipText);
    });
    
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        linkLabels
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2 - 8);
        
        nodeGroup
            .attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function showAbout() {
    alert('🏠 House Price Prediction AI\n\n' +
          'Hệ thống dự đoán giá nhà sử dụng Machine Learning\n' +
          'Dữ liệu từ Batdongsan.vn\n' +
          'Model: Random Forest Regressor\n' +
          '© 2026 - Intelligent System Development');
}