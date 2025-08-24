// script.js (新版本：支持多语言 + 移动版)
document.addEventListener('DOMContentLoaded', () => {

    // --- 【重要】请在这里配置你的后端API地址 ---
    const BACKEND_API_URL = 'https://jeffliulab.com';
    
    // --- 获取DOM元素 ---
    const reportContentEl = document.getElementById('report-content');
    const sidebarContentEl = document.getElementById('sidebar-content');
    const langSwitcherEl = document.getElementById('lang-switcher');
    // --- NEW: Get mobile-specific elements ---
    const menuToggle = document.getElementById('menu-toggle');
    const mobileOverlay = document.getElementById('mobile-overlay');
    const sidebarEl = document.getElementById('sidebar'); // We need the sidebar itself

    // --- 状态管理 ---
    let allReportsData = {}; // 用于存储从后端获取的完整数据树
    let currentLang = 'en';  // 当前语言，默认为英语

    // --- 语言翻译文本 ---
    const translations = {
        en: {
            title: "Fincancial News Agent",
            history_title: "Historical Reports",
            loading_latest: "Loading latest report...",
            loading_list: "Loading list...",
            loading_report: "Loading report...",
            no_reports_found: "No reports found. Please generate reports in the backend first.",
            load_failed: "Failed to load. Please check if the backend service is running, or view the console for errors."
        },
        cn: {
            title: "金融新闻智能体",
            history_title: "历史报告",
            loading_latest: "正在加载最新报告...",
            loading_list: "正在加载列表...",
            loading_report: "正在加载报告...",
            no_reports_found: "没有找到任何报告，请先在后端生成报告。",
            load_failed: "加载失败。请检查后端服务是否正常运行，或查看控制台错误。"
        }
    };

    // --- 函数 ---

    // 更新页面上的所有UI文本
    function translateUI(lang) {
        document.querySelectorAll('[data-translate]').forEach(el => {
            const key = el.dataset.translate;
            if (translations[lang] && translations[lang][key]) {
                el.textContent = translations[lang][key];
            }
        });
    }

    // 根据文件名获取并显示报告
    async function fetchAndDisplayReport(filename) {
        reportContentEl.innerHTML = `<div class="loader"></div><p>${translations[currentLang].loading_report}</p>`;
        try {
            const response = await fetch(`${BACKEND_API_URL}/api/reports/${filename}`);
            if (!response.ok) throw new Error(`Network error: ${response.statusText}`);
            const markdownText = await response.text();
            reportContentEl.innerHTML = marked.parse(markdownText);
        } catch (error) {
            console.error('获取报告失败:', error);
            reportContentEl.innerHTML = `<p style="color:red;">${translations[currentLang].load_failed}</p>`;
        }
    }
    
    // 根据当前语言构建侧边栏
    function buildSidebar(lang) {
        sidebarContentEl.innerHTML = '';
        const tree = allReportsData[lang] || {};

        const monthNames = {
            "01": "January", "02": "February", "03": "March", "04": "April",
            "05": "May", "06": "June", "07": "July", "08": "August",
            "09": "September", "10": "October", "11": "November", "12": "December"
        };

        if (Object.keys(tree).length === 0) {
            sidebarContentEl.innerHTML = `<p>${translations[lang].no_reports_found}</p>`;
            return;
        }

        const years = Object.keys(tree).sort((a, b) => b - a);
        years.forEach(year => {
            const yearContainer = document.createElement('div');
            const yearToggle = document.createElement('div');
            yearToggle.className = 'year-toggle';
            yearToggle.textContent = year;
            yearContainer.appendChild(yearToggle);
            
            const monthList = document.createElement('ul');
            monthList.className = 'month-list';
            yearContainer.appendChild(monthList);
            
            const months = Object.keys(tree[year]).sort((a, b) => b - a);
            months.forEach(month => {
                const monthItem = document.createElement('li');
                const monthToggle = document.createElement('div');
                monthToggle.className = 'month-toggle';
                
                if (lang === 'cn') {
                    monthToggle.textContent = `${month}月`;
                } else {
                    monthToggle.textContent = monthNames[month] || month;
                }

                monthItem.appendChild(monthToggle);
                
                const reportList = document.createElement('ul');
                reportList.className = 'report-list';
                monthItem.appendChild(reportList);
                
                tree[year][month].forEach(filename => {
                    const reportItem = document.createElement('li');
                    const link = document.createElement('a');
                    link.href = '#';
                    link.className = 'report-link';
                    link.textContent = filename.replace(/NR_|_CN\.md|\.md/g, '');
                    link.dataset.filename = filename;
                    reportItem.appendChild(link);
                    reportList.appendChild(reportItem);
                });
                monthList.appendChild(monthItem);
            });
            sidebarContentEl.appendChild(yearContainer);
        });
    }

    // 加载指定语言的最新报告
    function loadLatestReport(lang) {
        const langData = allReportsData[lang] || {};
        const years = Object.keys(langData).sort((a,b)=>b-a);
        if (years.length > 0) {
            const latestYear = years[0];
            const months = Object.keys(langData[latestYear]).sort((a,b)=>b-a);
            if (months.length > 0) {
                const latestMonth = months[0];
                const latestFilename = langData[latestYear][latestMonth][0];
                if (latestFilename) {
                    fetchAndDisplayReport(latestFilename);
                    setTimeout(() => { 
                        const firstLink = sidebarContentEl.querySelector('.report-link');
                        if(firstLink) {
                            firstLink.classList.add('active');
                            firstLink.closest('.report-list').classList.add('open');
                            firstLink.closest('.month-list').classList.add('open');
                            firstLink.closest('.report-list').previousElementSibling.classList.add('open');
                            firstLink.closest('.month-list').previousElementSibling.classList.add('open');
                        }
                    }, 100);
                    return;
                }
            }
        }
        reportContentEl.innerHTML = `<p>${translations[lang].no_reports_found}</p>`;
    }

    // --- 事件监听 ---

    // 侧边栏点击事件
    sidebarContentEl.addEventListener('click', (e) => {
        e.preventDefault();
        const target = e.target;
        if (target.classList.contains('year-toggle') || target.classList.contains('month-toggle')) {
            target.classList.toggle('open');
            target.nextElementSibling?.classList.toggle('open');
        }
        if (target.classList.contains('report-link')) {
            document.querySelector('.report-link.active')?.classList.remove('active');
            target.classList.add('active');
            fetchAndDisplayReport(target.dataset.filename);
            
            // --- NEW: Close mobile sidebar after selecting a report ---
            document.body.classList.remove('sidebar-open');
        }
    });

    // 语言切换器点击事件
    langSwitcherEl.addEventListener('click', (e) => {
        const target = e.target.closest('button');
        if (target) {
            const lang = target.dataset.lang;
            if (lang !== currentLang) {
                currentLang = lang;
                document.querySelector('#lang-switcher button.active').classList.remove('active');
                target.classList.add('active');
                
                translateUI(currentLang);
                buildSidebar(currentLang);
                loadLatestReport(currentLang);
            }
        }
    });

    // --- NEW: Mobile Menu Event Listeners ---
    menuToggle.addEventListener('click', () => {
        document.body.classList.toggle('sidebar-open');
    });

    mobileOverlay.addEventListener('click', () => {
        document.body.classList.remove('sidebar-open');
    });


    // --- 页面初始化 ---
    async function initializePage() {
        try {
            const response = await fetch(`${BACKEND_API_URL}/api/reports`);
            if (!response.ok) throw new Error('无法获取报告列表');
            
            allReportsData = await response.json();
            
            translateUI(currentLang);
            buildSidebar(currentLang);
            loadLatestReport(currentLang);

        } catch (error) {
            console.error('页面初始化失败:', error);
            reportContentEl.innerHTML = `<p style="color:red;">${translations.en.load_failed}</p>`;
            sidebarContentEl.innerHTML = `<p style="color:red;">Failed to load list.</p>`;
        }
    }

    initializePage();
});