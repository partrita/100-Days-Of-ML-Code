// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="index.html"><strong aria-hidden="true">1.</strong> Introduction</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day 1_Data PreProcessing.html"><strong aria-hidden="true">2.</strong> 데이터 전처리 | 1일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day2_Simple_Linear_Regression.html"><strong aria-hidden="true">3.</strong> 단순 선형 회귀 | 2일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day3_Multiple_Linear_Regression.html"><strong aria-hidden="true">4.</strong> 다중 선형 회귀 | 3일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day 6 Logistic Regression.html"><strong aria-hidden="true">5.</strong> 로지스틱 회귀 구현 | 6일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day 11 K-NN.html"><strong aria-hidden="true">6.</strong> K-NN 구현 | 11일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day 13 SVM.html"><strong aria-hidden="true">7.</strong> SVM 구현 | 14일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day 25 Decision Tree.html"><strong aria-hidden="true">8.</strong> 결정 트리 구현 | 25일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day 34 Random_Forest.html"><strong aria-hidden="true">9.</strong> 랜덤 포레스트 구현 | 34일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_55_Reinforcement_Learning_Intro.html"><strong aria-hidden="true">10.</strong> 강화 학습 소개 | 55일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_56_Markov_Decision_Processes.html"><strong aria-hidden="true">11.</strong> 마르코프 결정 과정 | 56일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_57_Q_Learning_Theory_Algorithm.html"><strong aria-hidden="true">12.</strong> Q-러닝 - 이론 및 알고리즘 | 57일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_58_Q_Learning_Simple_Implementation.html"><strong aria-hidden="true">13.</strong> Q-러닝 - 간단한 구현 | 58일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_59_Deep_Q_Networks_DQN.html"><strong aria-hidden="true">14.</strong> 심층 Q-네트워크 (DQN) | 59일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_60_Natural_Language_Processing_Intro.html"><strong aria-hidden="true">15.</strong> 자연어 처리 소개 | 60일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_61_Text_Preprocessing.html"><strong aria-hidden="true">16.</strong> 텍스트 전처리 | 61일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_62_Bag_of_Words_TF-IDF.html"><strong aria-hidden="true">17.</strong> Bag-of-Words 및 TF-IDF | 62일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_63_Word_Embeddings_Word2Vec_GloVe_Conceptual.html"><strong aria-hidden="true">18.</strong> 단어 임베딩 - Word2Vec, GloVe (개념) | 63일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_64_Implementing_Word2Vec_Gensim.html"><strong aria-hidden="true">19.</strong> Word2Vec 구현 | 64일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_65_Sentiment_Analysis_Basic_Techniques.html"><strong aria-hidden="true">20.</strong> 감성 분석 - 기본 기법 | 65일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_66_Building_Simple_Sentiment_Analyzer.html"><strong aria-hidden="true">21.</strong> 간단한 감성 분석기 구축 | 66일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_67_Introduction_to_Recurrent_Neural_Networks_RNNs.html"><strong aria-hidden="true">22.</strong> 순환 신경망 (RNN) 소개 | 67일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_68_Long_Short_Term_Memory_LSTM_Networks.html"><strong aria-hidden="true">23.</strong> LSTM (Long Short-Term Memory) 네트워크 | 68일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_69_Building_Text_Classifier_RNN_LSTM.html"><strong aria-hidden="true">24.</strong> RNN/LSTM을 이용한 텍스트 분류기 구축 | 69일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_70_Review_RL_NLP_Concepts.html"><strong aria-hidden="true">25.</strong> 강화 학습 및 NLP 개념 복습 | 70일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_71_PCA_Theory_Implementation.html"><strong aria-hidden="true">26.</strong> 주성분 분석 (PCA) - 이론 및 구현 | 71일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_72_Linear_Discriminant_Analysis_LDA.html"><strong aria-hidden="true">27.</strong> 선형 판별 분석 (LDA) | 72일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_73_Model_Evaluation_Metrics.html"><strong aria-hidden="true">28.</strong> 모델 평가 지표 | 73일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_74_Cross_Validation_Techniques.html"><strong aria-hidden="true">29.</strong> 교차 검증 기법 | 74일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_75_Hyperparameter_Tuning_Grid_Random_Search.html"><strong aria-hidden="true">30.</strong> 하이퍼파라미터 튜닝 | 75일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_76_Ensemble_Methods_Bagging_Boosting.html"><strong aria-hidden="true">31.</strong> 앙상블 방법 - 배깅, 부스팅 | 76일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_77_XGBoost_Introduction_Implementation.html"><strong aria-hidden="true">32.</strong> XGBoost - 소개 및 구현 | 77일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_78_LightGBM_Introduction_Implementation.html"><strong aria-hidden="true">33.</strong> LightGBM - 소개 및 구현 | 78일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_79_Introduction_to_Time_Series_Analysis.html"><strong aria-hidden="true">34.</strong> 시계열 분석 소개 | 79일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_80_ARIMA_Models_for_Time_Series.html"><strong aria-hidden="true">35.</strong> 시계열을 위한 ARIMA 모델 | 80일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_81_Introduction_to_Model_Deployment.html"><strong aria-hidden="true">36.</strong> 모델 배포 소개 | 81일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_82_Flask_Django_for_ML_Deployment_Basics.html"><strong aria-hidden="true">37.</strong> ML 모델 배포를 위한 Flask/Django - 기초 | 82일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_83_Creating_Simple_API_for_ML_Model.html"><strong aria-hidden="true">38.</strong> ML 모델을 위한 간단한 API 만들기 | 83일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_84_Docker_for_ML_Basics.html"><strong aria-hidden="true">39.</strong> ML을 위한 Docker - 기초 | 84일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_85_Deploying_Simple_ML_Model_Docker.html"><strong aria-hidden="true">40.</strong> Docker를 사용한 간단한 ML 모델 배포 | 85일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_86_Brainstorming_Capstone_Project_Ideas.html"><strong aria-hidden="true">41.</strong> 캡스톤 프로젝트 아이디어 브레인스토밍 | 86일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_87_Selecting_Capstone_Project_Scope_Definition.html"><strong aria-hidden="true">42.</strong> 캡스톤 프로젝트 선정 및 범위 정의 | 87일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_88_Data_Collection_Preprocessing_Capstone.html"><strong aria-hidden="true">43.</strong> 프로젝트를 위한 데이터 수집 및 전처리 | 88일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_89_EDA_Feature_Engineering_Capstone.html"><strong aria-hidden="true">44.</strong> 프로젝트를 위한 탐색적 데이터 분석 (EDA) | 89일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_90_Model_Selection_Initial_Training_Capstone.html"><strong aria-hidden="true">45.</strong> 프로젝트를 위한 모델 선정 및 초기 학습 | 90일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_91_Model_Evaluation_Iteration_Capstone.html"><strong aria-hidden="true">46.</strong> 프로젝트를 위한 모델 평가 및 반복 | 91일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_92_Fine_tuning_Optimization_Capstone.html"><strong aria-hidden="true">47.</strong> 프로젝트를 위한 미세 조정 및 최적화 | 92일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_93_Building_UI_Presentation_Capstone.html"><strong aria-hidden="true">48.</strong> 프로젝트를 위한 간단한 UI 또는 프레젠테이션 구축 | 93일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_94_Documenting_Capstone_Project.html"><strong aria-hidden="true">49.</strong> 캡스톤 프로젝트 문서화 | 94일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_95_Introduction_to_GANs.html"><strong aria-hidden="true">50.</strong> 생성적 적대 신경망 (GAN) 소개 | 95일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_96_Autoencoders.html"><strong aria-hidden="true">51.</strong> 오토인코더 | 96일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_97_Explainable_AI_XAI_LIME_SHAP.html"><strong aria-hidden="true">52.</strong> 설명 가능한 AI (XAI) - LIME, SHAP | 97일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_98_Ethics_in_AI_ML.html"><strong aria-hidden="true">53.</strong> AI 및 머신러닝 윤리 | 98일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_99_Future_Trends_in_ML_AI.html"><strong aria-hidden="true">54.</strong> ML 및 AI의 미래 동향 | 99일차</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="Code/Day_100_Review_100_Days_Next_Steps.html"><strong aria-hidden="true">55.</strong> 100일 여정 복습 및 다음 단계 | 100일차</a></span></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            if (link.href === current_page
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level > level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

