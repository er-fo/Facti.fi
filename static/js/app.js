document.addEventListener('DOMContentLoaded', function() {
    console.log('JavaScript loaded and DOM ready');
    
    const form = document.getElementById('analyzeForm');
    const urlInput = document.getElementById('urlInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loading');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const progressSection = document.getElementById('progress-section');

    // Debug: Check if all elements are found
    console.log('Form element:', form);
    console.log('URL input:', urlInput);
    console.log('Analyze button:', analyzeBtn);
    console.log('Button text:', btnText);
    console.log('Button loader:', btnLoader);

    // Progress elements
    const progressBarFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-message');
    const progressPercentage = document.getElementById('progress-percentage');
    const stepElements = {
        extraction: document.getElementById('step-extraction'),
        transcription: document.getElementById('step-transcription'),
        analysis: document.getElementById('step-analysis'),
        research: document.getElementById('step-research')
    };

    let currentRequestId = null;
    let progressInterval = null;

    // Platform toggle functionality
    const platformToggle = document.getElementById('platformToggle');
    const platformList = document.getElementById('platformList');
    
    platformToggle.addEventListener('click', function() {
        const isVisible = platformList.style.display !== 'none';
        platformList.style.display = isVisible ? 'none' : 'block';
        platformToggle.textContent = isVisible ? '📋 View Supported Platforms' : '📋 Hide Supported Platforms';
    });

    // Add event listener with error checking
    if (form) {
        console.log('Adding submit event listener to form');
        form.addEventListener('submit', handleAnalysis);
    } else {
        console.error('Form element not found!');
    }

    async function handleAnalysis(e) {
        console.log('handleAnalysis called', e);
        e.preventDefault();
        
        const url = urlInput.value.trim();
        console.log('URL from input:', url);
        if (!url) {
            showError('Please enter a valid URL');
            return;
        }

        // Show loading state and progress
        setLoadingState(true);
        hideError();
        hideResults();
        showProgress();

        try {
            // Get selected prompt type
            const promptType = document.querySelector('input[name="analysisType"]:checked')?.value || 'comprehensive';
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    url: url,
                    prompt_type: promptType
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Analysis request failed');
            }

            if (data.status === 'processing') {
                // Start polling for progress and results
                pollProgress(data.request_id);
            } else if (data.status === 'completed') {
                // Analysis completed immediately
                displayResults(data);
                hideProgress();
            } else {
                throw new Error('Unexpected response status: ' + data.status);
            }

        } catch (error) {
            console.error('Analysis error:', error);
            showError('Analysis failed: ' + error.message);
            hideProgress();
        } finally {
            setLoadingState(false);
        }
    }

    async function pollProgress(requestId) {
        const pollInterval = 1000; // Poll every second
        let attempts = 0;
        const maxAttempts = 300; // 5 minutes maximum

        const poll = async () => {
            attempts++;
            
            try {
                // Check progress
                const progressResponse = await fetch(`/progress/${requestId}`);
                const progressData = await progressResponse.json();
                
                if (progressData.percentage !== undefined) {
                    updateProgress(progressData.percentage, progressData.message || 'Processing...');
                    
                    // Check for error states in progress data
                    if (progressData.step === 'error') {
                        throw new Error(progressData.message || 'Analysis failed');
                    }
                }

                // Check if analysis is complete
                const resultsResponse = await fetch(`/results/${requestId}`);
                
                if (resultsResponse.ok) {
                    const resultsData = await resultsResponse.json();
                    
                    if (resultsData.status === 'completed') {
                        displayResults(resultsData);
                        hideProgress();
                        return; // Stop polling
                    } else if (resultsData.status === 'error') {
                        // Pass the full error data to displayResults for proper error handling
                        displayResults(resultsData);
                        hideProgress();
                        return; // Stop polling
                    }
                }

                // Continue polling if not complete and haven't exceeded max attempts
                if (attempts < maxAttempts) {
                    setTimeout(poll, pollInterval);
                } else {
                    throw new Error('Analysis timed out - please try again');
                }

            } catch (error) {
                console.error('Polling error:', error);
                showError('Analysis failed: ' + error.message);
                hideProgress();
            }
        };

        poll();
    }

    function startProgressPolling(requestId) {
        if (progressInterval) {
            clearInterval(progressInterval);
        }

        progressInterval = setInterval(async () => {
            try {
                // Poll for progress
                const progressResponse = await fetch(`/progress/${requestId}`);
                if (progressResponse.ok) {
                    const progressData = await progressResponse.json();
                    updateProgress(progressData.percentage, progressData.message || 'Processing...');
                    
                    // Check if analysis is complete
                    if (progressData.step === 'complete' && progressData.percentage === 100) {
                        // Get final results
                        const resultsResponse = await fetch(`/results/${requestId}`);
                        if (resultsResponse.ok) {
                            const resultsData = await resultsResponse.json();
                            if (resultsData.status === 'completed') {
                                displayResults(resultsData);
                                setLoadingState(false);
                                hideProgress();
                                stopProgressPolling();
                            } else if (resultsData.status === 'error') {
                                displayResults(resultsData);
                                setLoadingState(false);
                                hideProgress();
                                stopProgressPolling();
                            }
                        }
                    }
                } else {
                    // Progress endpoint returned error status, check results endpoint
                    console.log('Progress request failed, checking results...');
                    
                    // Try to get results
                    const resultsResponse = await fetch(`/results/${requestId}`);
                    if (resultsResponse.ok) {
                        const resultsData = await resultsResponse.json();
                        if (resultsData.status === 'completed') {
                            displayResults(resultsData);
                            setLoadingState(false);
                            hideProgress();
                            stopProgressPolling();
                        } else if (resultsData.status === 'error') {
                            displayResults(resultsData);
                            setLoadingState(false);
                            hideProgress();
                            stopProgressPolling();
                        } else if (resultsData.status === 'processing') {
                            // Still processing, continue polling
                            console.log('Analysis still in progress...');
                        } else if (resultsData.status === 'unknown') {
                            // Request may have timed out or failed
                            showError('Analysis request timed out or failed. Please try again.');
                            setLoadingState(false);
                            hideProgress();
                            stopProgressPolling();
                        }
                    } else {
                        // Results endpoint also failed - log but don't stop polling immediately
                        console.log('Both progress and results endpoints failed, will retry...');
                        // Continue polling for a bit longer in case of temporary network issues
                    }
                }
            } catch (error) {
                console.error('Progress polling error:', error);
                // Don't stop polling immediately on network errors - keep trying
            }
        }, 500); // Poll every 500ms
    }

    function stopProgressPolling() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    function updateProgress(percentage, message) {
        // Update progress bar
        progressBarFill.style.width = `${percentage}%`;
        progressText.textContent = message;
        progressPercentage.textContent = `${percentage}%`;

        // Update step indicators
        updateStepIndicators(percentage);
    }

    function updateStepIndicators(percentage) {
        // Reset all steps
        Object.values(stepElements).forEach(el => {
            el.classList.remove('active', 'completed');
        });

        // Map steps to elements
        const stepMap = {
            extraction: 'extraction',
            transcription: 'transcription', 
            analysis: 'analysis',
            research: 'research',
            cleanup: 'research', // Map cleanup to research step
            complete: 'research'
        };

        const mappedStep = stepMap[percentage === 100 ? 'complete' : 'extraction'];
        if (!mappedStep) return;

        // Mark previous steps as completed
        const stepOrder = ['extraction', 'transcription', 'analysis', 'research'];
        const currentStepIndex = stepOrder.indexOf(mappedStep);
        
        for (let i = 0; i < currentStepIndex; i++) {
            stepElements[stepOrder[i]].classList.add('completed');
        }

        // Mark current step as active (unless it's complete)
        if (percentage < 100 || percentage === 100) {
            stepElements[mappedStep].classList.add(percentage === 100 ? 'completed' : 'active');
        }
    }

    function showProgress() {
        progressSection.style.display = 'block';
        progressSection.scrollIntoView({ behavior: 'smooth' });
        
        // Reset progress state
        progressBarFill.style.width = '0%';
        progressText.textContent = 'Initializing...';
        progressPercentage.textContent = '0%';
        
        // Reset steps
        Object.values(stepElements).forEach(el => {
            el.classList.remove('active', 'completed');
        });
    }

    function hideProgress() {
        progressSection.style.display = 'none';
    }

    function setLoadingState(loading) {
        if (loading) {
            analyzeBtn.disabled = true;
            btnText.style.display = 'none';
            btnLoader.style.display = 'inline';
            urlInput.disabled = true;
        } else {
            analyzeBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
            urlInput.disabled = false;
        }
    }

    function displayResults(data) {
        // Check if this is an error result
        if (data.status === 'error') {
            showError(data.error || 'Analysis failed with unknown error');
            hideProgress();
            return;
        }
        
        // Check if analysis data is present
        if (!data.analysis) {
            showError('Analysis data is missing from the results');
            hideProgress();
            return;
        }
        
        // Hide error and progress, show results
        hideError();
        hideProgress();
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Display content info
        document.getElementById('content-title').textContent = data.title || 'Unknown Title';
        document.getElementById('analysis-timestamp').textContent = `Analyzed on: ${formatTimestamp(data.timestamp)}`;

        // Display credibility score with fallback
        const credibilityScore = data.analysis.credibility_score || data.analysis.truthfulness?.overall_score || 50;
        displayCredibilityScore(credibilityScore);

        // Check analysis type and display appropriate sections
        const promptType = data.analysis.prompt_type || 'comprehensive';
        console.log('Analysis prompt type:', promptType);
        
        if (promptType === 'simple') {
            displaySimpleAnalysis(data.analysis);
        } else if (data.analysis.enhanced_analysis && data.analysis.analysis_type === 'political_rhetoric') {
            displayEnhancedAnalysis(data.analysis);
        }

        // Display standard analysis details with safe access
        displayKeyClaimsList('key-claims', data.analysis.key_claims || []);
        displayKeyClaimsList('red-flags', data.analysis.red_flags || []);
        document.getElementById('factual-accuracy').textContent = data.analysis.factual_accuracy || 'Not available';
        displayKeyClaimsList('bias-indicators', data.analysis.bias_indicators || []);
        document.getElementById('evidence-quality').textContent = data.analysis.evidence_quality || 'Not available';
        document.getElementById('analysis-summary').textContent = data.analysis.analysis_summary || 'Not available';
        
        // Display chain of thought reasoning
        const chainOfThought = data.analysis.chain_of_thought || 'Chain of thought reasoning not available for this analysis.';
        document.getElementById('chain-of-thought').textContent = chainOfThought;

        // Display transcript - use enhanced format if available, fallback to full_text
        if (data.transcript && typeof data.transcript === 'object') {
            const transcriptContent = data.transcript.enhanced_display || data.transcript.full_text || 'No transcript available';
            document.getElementById('transcript-content').textContent = transcriptContent;
        } else {
            document.getElementById('transcript-content').textContent = data.transcript || 'No transcript available';
        }

        // Display research results
        displayResearchResults(data.research);
        
        // Show video generation section and set up handlers
        showVideoGenerationSection(data.request_id);
    }

    function displayEnhancedAnalysis(analysis) {
        // Show enhanced sections
        const enhancedSections = document.getElementById('enhanced-sections');
        if (enhancedSections) {
            enhancedSections.style.display = 'block';
        }

        // Check and display each section only if it has meaningful data
        if (hasMeaningfulRhetoricalTactics(analysis.rhetorical_tactics)) {
            displayRhetoricalTactics(analysis.rhetorical_tactics || []);
        } else {
            hideSection('rhetorical-tactics-section');
        }
        
        if (hasMeaningfulHateSpeech(analysis.hate_speech)) {
            displayHateSpeechAnalysis(analysis.hate_speech || {});
        } else {
            hideSection('hate-speech-section');
        }
        
        if (hasMeaningfulPsychologicalMarkers(analysis.psychological_markers)) {
            displayPsychologicalMarkers(analysis.psychological_markers || {});
        } else {
            hideSection('psychological-markers-section');
        }
        
        if (hasMeaningfulContradictions(analysis.contradictions)) {
            displayContradictions(analysis.contradictions || {});
        } else {
            hideSection('contradictions-section');
        }
        
        if (hasMeaningfulExtortion(analysis.extortion_tendencies)) {
            displayExtortionAnalysis(analysis.extortion_tendencies || {});
        } else {
            hideSection('extortion-section');
        }
        
        if (hasMeaningfulSubjectiveClaims(analysis.subjective_claims)) {
            displaySubjectiveClaims(analysis.subjective_claims || []);
        } else {
            hideSection('subjective-claims-section');
        }
    }

    function displaySimpleAnalysis(analysis) {
        console.log('🔄 Displaying simple analysis');
        
        try {
            // Hide enhanced sections and show simple structure
            const enhancedSections = document.getElementById('enhanced-sections');
            if (enhancedSections) {
                enhancedSections.style.display = 'none';
                console.log('✅ Enhanced sections hidden');
            }
            
            // Create simple analysis structure with safer container detection
            let detailsContainer = document.querySelector('.analysis-details');
            
            // If .analysis-details doesn't exist, try alternative containers
            if (!detailsContainer) {
                console.warn('⚠️ .analysis-details container not found, trying alternatives...');
                detailsContainer = document.getElementById('results-section') || 
                                 document.querySelector('.results-container') ||
                                 document.querySelector('main');
                
                if (!detailsContainer) {
                    throw new Error('No suitable container found for simple analysis display');
                }
                console.log('✅ Using alternative container:', detailsContainer.className || detailsContainer.id);
            }
            
            // Remove existing simple analysis sections if they exist
            const existingSimple = detailsContainer.querySelector('.simple-analysis-sections');
            if (existingSimple) {
                existingSimple.remove();
                console.log('✅ Removed existing simple analysis sections');
            }
            
            // Create new simple analysis sections with error handling
            const simpleContainer = document.createElement('div');
            simpleContainer.className = 'simple-analysis-sections';
            
            // Build the HTML structure for simple analysis
            const simpleHTML = `
                <div class="dropdown-section key-claims-section">
                    <div class="dropdown-header" onclick="toggleDropdown('simple-key-claims')">
                        <div class="dropdown-title">
                            <span class="dropdown-icon">🔍</span>
                            <h4>Key Claims</h4>
                        </div>
                        <div class="dropdown-arrow">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6,9 12,15 18,9"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div class="dropdown-content" id="simple-key-claims" style="display: block;">
                        <div class="claims-container"></div>
                    </div>
                </div>
                
                <div class="dropdown-section red-flags-section">
                    <div class="dropdown-header" onclick="toggleDropdown('simple-red-flags')">
                        <div class="dropdown-title">
                            <span class="dropdown-icon">🚩</span>
                            <h4>Red Flags</h4>
                        </div>
                        <div class="dropdown-arrow">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6,9 12,15 18,9"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div class="dropdown-content" id="simple-red-flags" style="display: block;">
                        <div class="red-flags-container"></div>
                    </div>
                </div>
                
                <div class="dropdown-section overall-assessment-section">
                    <div class="dropdown-header" onclick="toggleDropdown('simple-assessment')">
                        <div class="dropdown-title">
                            <span class="dropdown-icon">⚖️</span>
                            <h4>Overall Assessment</h4>
                        </div>
                        <div class="dropdown-arrow">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6,9 12,15 18,9"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div class="dropdown-content" id="simple-assessment" style="display: block;">
                        <div class="assessment-container"></div>
                    </div>
                </div>
                
                <div class="dropdown-section simple-summary-section">
                    <div class="dropdown-header" onclick="toggleDropdown('simple-summary')">
                        <div class="dropdown-title">
                            <span class="dropdown-icon">📝</span>
                            <h4>Summary</h4>
                        </div>
                        <div class="dropdown-arrow">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6,9 12,15 18,9"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div class="dropdown-content" id="simple-summary" style="display: block;">
                        <div class="summary-container"></div>
                    </div>
                </div>
            `;
            
            simpleContainer.innerHTML = simpleHTML;
            console.log('✅ Simple analysis HTML structure created');
            
            // Insert the simple analysis structure with error handling
            try {
                if (detailsContainer.firstChild) {
                    detailsContainer.insertBefore(simpleContainer, detailsContainer.firstChild);
                } else {
                    detailsContainer.appendChild(simpleContainer);
                }
                console.log('✅ Simple analysis structure inserted into DOM');
            } catch (insertError) {
                console.error('❌ Error inserting simple analysis structure:', insertError);
                detailsContainer.appendChild(simpleContainer); // Fallback to append
            }
            
            // Populate the sections with data, with comprehensive error handling
            try {
                console.log('🔄 Populating simple analysis sections...');
                
                // Extract data from analysis with proper mapping for simple analysis format
                const keyClaims = analysis.subjective_claims || analysis.key_claims || [];
                const redFlags = analysis.contradictions || analysis.red_flags || [];
                const assessment = analysis.truthfulness || analysis.overall_assessment || {};
                const summary = analysis.key_takeaways || analysis.speech_takeaway_summary || analysis.summary || {};
                
                console.log('📊 Data for simple analysis:', {
                    keyClaimsCount: Array.isArray(keyClaims) ? keyClaims.length : 'not_array',
                    redFlagsCount: Array.isArray(redFlags) ? redFlags.length : 'not_array',
                    hasAssessment: !!assessment,
                    hasSummary: !!summary
                });
                
                displaySimpleKeyClaims(keyClaims);
                displaySimpleRedFlags(redFlags);
                displaySimpleAssessment(assessment);
                displaySimpleSummary(summary);
                
                console.log('✅ Simple analysis sections populated successfully');
            } catch (populateError) {
                console.error('❌ Error populating simple analysis sections:', populateError);
                // Show a fallback message instead of failing completely
                const fallbackDiv = document.createElement('div');
                fallbackDiv.style.cssText = 'padding: 20px; text-align: center; color: #666;';
                fallbackDiv.innerHTML = `
                    <p>⚠️ There was an issue displaying the detailed analysis sections.</p>
                    <p>The analysis data is available in the standard sections below.</p>
                `;
                simpleContainer.appendChild(fallbackDiv);
            }
            
        } catch (error) {
            console.error('❌ Critical error in displaySimpleAnalysis:', error);
            // Don't throw the error - let the main displayResults continue with standard sections
            console.log('🔄 Falling back to standard analysis display');
        }
    }

    // Helper functions to check if sections have meaningful data
    function hasMeaningfulRhetoricalTactics(tactics) {
        return tactics && tactics.length > 0 && tactics.some(tactic => tactic.occurrences > 0);
    }

    function hasMeaningfulHateSpeech(hateSpeech) {
        return hateSpeech && hateSpeech.overall && hateSpeech.overall.occurrences > 0;
    }

    function hasMeaningfulPsychologicalMarkers(markers) {
        if (!markers) return false;
        const markerTypes = ['narcissism', 'authoritarianism', 'fascism_rhetoric'];
        return markerTypes.some(type => markers[type] && markers[type].score > 0);
    }

    function hasMeaningfulContradictions(contradictions) {
        return contradictions && contradictions.score > 0;
    }

    function hasMeaningfulExtortion(extortion) {
        return extortion && extortion.score > 0;
    }

    function hasMeaningfulSubjectiveClaims(claims) {
        return claims && claims.length > 0;
    }

    function hideSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'none';
        }
    }

    function showSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'block';
        }
    }

    function createDropdownContainer(title, contentId, icon = '') {
        return `
            <div class="dropdown-section" id="${contentId}-section">
                <div class="dropdown-header" onclick="toggleDropdown('${contentId}')">
                    <div class="dropdown-title">
                        <span class="dropdown-icon">${icon}</span>
                        <h4>${title}</h4>
                    </div>
                    <div class="dropdown-arrow">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6,9 12,15 18,9"></polyline>
                        </svg>
                    </div>
                </div>
                <div class="dropdown-content" id="${contentId}" style="display: none;">
                    <!-- Content will be populated by specific display functions -->
                </div>
            </div>
        `;
    }

    function toggleDropdown(contentId) {
        const content = document.getElementById(contentId);
        const arrow = content.parentElement.querySelector('.dropdown-arrow svg');
        
        if (content.style.display === 'none') {
            content.style.display = 'block';
            arrow.style.transform = 'rotate(180deg)';
        } else {
            content.style.display = 'none';
            arrow.style.transform = 'rotate(0deg)';
        }
    }

    function displayRhetoricalTactics(tactics) {
        const container = document.getElementById('rhetorical-tactics').querySelector('.tactics-grid');
        container.innerHTML = '';

        if (!tactics || tactics.length === 0) {
            container.innerHTML = '<p style="color: #718096; font-style: italic;">No rhetorical tactics identified</p>';
            return;
        }

        tactics.forEach(tactic => {
            const tacticDiv = document.createElement('div');
            tacticDiv.className = 'tactic-item';
            tacticDiv.style.cssText = `
                margin-bottom: 1rem;
                padding: 1rem;
                background: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            `;

            const intensityColor = tactic.intensity_score > 0.7 ? '#e53e3e' : 
                                 tactic.intensity_score > 0.4 ? '#ed8936' : '#38a169';

            tacticDiv.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h5 style="color: #2d3748; margin: 0; font-weight: 600;">${tactic.tactic}</h5>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #718096; font-size: 0.875rem;">${tactic.occurrences} occurrences</span>
                        <span style="color: ${intensityColor}; font-weight: 600;">${(tactic.intensity_score * 100).toFixed(0)}%</span>
                    </div>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <div style="background: #f7fafc; height: 4px; border-radius: 2px; overflow: hidden;">
                        <div style="height: 100%; background: ${intensityColor}; width: ${tactic.intensity_score * 100}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                <div style="color: #4a5568;">
                    ${tactic.examples && tactic.examples.length > 0 ? 
                        `<strong>Examples:</strong> ${tactic.examples.slice(0, 2).map(ex => `"${ex}"`).join(', ')}` :
                        'No examples available'
                    }
                </div>
            `;

            container.appendChild(tacticDiv);
        });
    }

    function displayHateSpeechAnalysis(hateSpeech) {
        const container = document.getElementById('hate-speech-analysis').querySelector('.hate-speech-grid');
        container.innerHTML = '';

        const categories = hateSpeech.by_category || {};
        const overall = hateSpeech.overall || {};

        // Overall summary
        const summaryDiv = document.createElement('div');
        summaryDiv.style.cssText = `
            padding: 1rem;
            background: ${overall.occurrences > 0 ? '#fffbfb' : '#f0fff4'};
            border: 1px solid ${overall.occurrences > 0 ? '#fed7d7' : '#c6f6d5'};
            border-radius: 8px;
            margin-bottom: 1rem;
        `;
        summaryDiv.innerHTML = `
            <h5 style="margin: 0 0 0.5rem 0; color: #2d3748;">Overall Assessment</h5>
            <p style="margin: 0; color: ${overall.occurrences > 0 ? '#c53030' : '#38a169'};">
                ${overall.occurrences || 0} instances detected | 
                Severity: ${((overall.severity_score || 0) * 100).toFixed(0)}%
            </p>
        `;
        container.appendChild(summaryDiv);

        // Category breakdown
        Object.entries(categories).forEach(([category, data]) => {
            const categoryDiv = document.createElement('div');
            categoryDiv.style.cssText = `
                margin-bottom: 0.75rem;
                padding: 0.75rem;
                background: white;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
            `;

            const severityColor = data.severity_score > 0.7 ? '#e53e3e' : 
                                 data.severity_score > 0.3 ? '#ed8936' : '#718096';

            categoryDiv.innerHTML = `
                <div style="display: flex; justify-content: between; align-items: center;">
                    <span style="color: #2d3748; font-weight: 500; text-transform: capitalize;">${category.replace('_', ' ')}</span>
                    <span style="color: ${severityColor}; font-size: 0.875rem;">
                        ${data.occurrences || 0} instances | ${((data.severity_score || 0) * 100).toFixed(0)}% severity
                    </span>
                </div>
            `;

            container.appendChild(categoryDiv);
        });
    }

    function displayPsychologicalMarkers(markers) {
        const container = document.getElementById('psychological-markers').querySelector('.psychological-grid');
        container.innerHTML = '';

        const markerTypes = ['narcissism', 'authoritarianism', 'fascism_rhetoric'];
        
        markerTypes.forEach(type => {
            const marker = markers[type];
            if (!marker) return;

            const markerDiv = document.createElement('div');
            markerDiv.style.cssText = `
                margin-bottom: 1rem;
                padding: 1rem;
                background: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            `;

            const score = marker.score || 0;
            const scoreColor = score > 0.7 ? '#e53e3e' : score > 0.4 ? '#ed8936' : '#38a169';

            markerDiv.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h5 style="color: #2d3748; margin: 0; font-weight: 600; text-transform: capitalize;">
                        ${type.replace('_', ' ')}
                    </h5>
                    <span style="color: ${scoreColor}; font-weight: 600;">${(score * 100).toFixed(0)}%</span>
                </div>
                <div style="background: #f7fafc; height: 4px; border-radius: 2px; overflow: hidden; margin-bottom: 0.5rem;">
                    <div style="height: 100%; background: ${scoreColor}; width: ${score * 100}%; transition: width 0.3s ease;"></div>
                </div>
                <p style="color: #4a5568; margin: 0; font-size: 0.875rem;">
                    ${marker.description || 'No description available'}
                </p>
            `;

            container.appendChild(markerDiv);
        });
    }

    function displayContradictions(contradictions) {
        const container = document.getElementById('contradictions-analysis');
        const score = contradictions.score || 0;
        const scoreColor = score > 0.7 ? '#e53e3e' : score > 0.4 ? '#ed8936' : '#38a169';

        container.innerHTML = `
            <div style="padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h5 style="color: #2d3748; margin: 0;">Contradiction Level</h5>
                    <span style="color: ${scoreColor}; font-weight: 600;">${(score * 100).toFixed(0)}%</span>
                </div>
                <p style="color: #4a5568; margin: 0 0 0.5rem 0; font-size: 0.875rem;">
                    ${contradictions.description || 'No contradictions identified'}
                </p>
                ${contradictions.examples && contradictions.examples.length > 0 ? 
                    `<div style="color: #718096; font-size: 0.875rem;">
                        <strong>Examples:</strong> ${contradictions.examples.slice(0, 2).join('; ')}
                    </div>` : ''
                }
            </div>
        `;
    }

    function displayExtortionAnalysis(extortion) {
        const container = document.getElementById('extortion-analysis');
        const score = extortion.score || 0;
        const scoreColor = score > 0.7 ? '#e53e3e' : score > 0.4 ? '#ed8936' : '#38a169';

        container.innerHTML = `
            <div style="padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h5 style="color: #2d3748; margin: 0;">Coercive Language Level</h5>
                    <span style="color: ${scoreColor}; font-weight: 600;">${(score * 100).toFixed(0)}%</span>
                </div>
                <p style="color: #4a5568; margin: 0 0 0.5rem 0; font-size: 0.875rem;">
                    ${extortion.description || 'No coercive language identified'}
                </p>
                ${extortion.examples && extortion.examples.length > 0 ? 
                    `<div style="color: #718096; font-size: 0.875rem;">
                        <strong>Examples:</strong> ${extortion.examples.slice(0, 2).join('; ')}
                    </div>` : ''
                }
            </div>
        `;
    }

    function displaySubjectiveClaims(claims) {
        const container = document.getElementById('subjective-claims').querySelector('ul');
        container.innerHTML = '';

        if (!claims || claims.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'No subjective claims identified';
            li.style.fontStyle = 'italic';
            li.style.color = '#718096';
            container.appendChild(li);
            return;
        }

        claims.forEach(claim => {
            const li = document.createElement('li');
            li.style.marginBottom = '0.5rem';
            
            const claimText = typeof claim === 'string' ? claim : claim.claim || JSON.stringify(claim);
            const frequency = typeof claim === 'object' ? claim.frequency : 1;
            
            li.innerHTML = `
                <div style="color: #2d3748;">${claimText}</div>
                ${frequency > 1 ? `<div style="color: #718096; font-size: 0.875rem;">Frequency: ${frequency}</div>` : ''}
            `;
            
            container.appendChild(li);
        });
    }

    function displayCredibilityScore(score) {
        const scoreElement = document.getElementById('score-value');
        const scoreDisplay = document.querySelector('.score-display');
        const scoreDescription = document.getElementById('score-description');

        // Handle invalid or missing scores
        if (score === undefined || score === null || isNaN(score)) {
            score = 50; // Default fallback
        }
        
        // Ensure score is a number and within valid range
        score = Math.max(0, Math.min(100, Number(score)));
        
        scoreElement.textContent = score;

        // Remove existing classes
        if (scoreDisplay) {
            scoreDisplay.classList.remove('high-credibility', 'medium-credibility', 'low-credibility');
        }

        // Add appropriate class and description based on score
        if (score >= 70) {
            if (scoreDisplay) scoreDisplay.classList.add('high-credibility');
            scoreDescription.textContent = 'High credibility score indicates the content appears to be largely truthful and well-supported.';
        } else if (score >= 40) {
            if (scoreDisplay) scoreDisplay.classList.add('medium-credibility');
            scoreDescription.textContent = 'Medium credibility score suggests mixed reliability. Some claims may need verification.';
        } else {
            if (scoreDisplay) scoreDisplay.classList.add('low-credibility');
            scoreDescription.textContent = 'Low credibility score indicates potential issues with accuracy or evidence quality.';
        }
    }

    function displayKeyClaimsList(elementId, items) {
        const element = document.getElementById(elementId);
        element.innerHTML = '';

        if (!items || items.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'None identified';
            li.style.fontStyle = 'italic';
            li.style.color = '#718096';
            element.appendChild(li);
            return;
        }

        items.forEach(item => {
            const li = document.createElement('li');
            
            // Handle both string and object formats
            if (typeof item === 'string') {
                li.textContent = item;
            } else if (typeof item === 'object' && item !== null) {
                // If it's an object, try to extract text content
                if (item.text) {
                    li.textContent = item.text;
                } else if (item.claim) {
                    li.textContent = item.claim;
                } else if (item.description) {
                    li.textContent = item.description;
                } else {
                    // Fallback: try to stringify meaningfully
                    const text = JSON.stringify(item);
                    li.textContent = text.length > 100 ? text.substring(0, 100) + '...' : text;
                }
                
                // Add timestamp if available
                if (item.timestamp) {
                    const timeSpan = document.createElement('span');
                    timeSpan.style.color = '#718096';
                    timeSpan.style.fontSize = '0.875rem';
                    timeSpan.style.marginLeft = '0.5rem';
                    timeSpan.textContent = `[${item.timestamp}]`;
                    li.appendChild(timeSpan);
                }
            } else {
                li.textContent = String(item);
            }
            
            element.appendChild(li);
        });
    }

    function displayResearchResults(research) {
        const researchElement = document.getElementById('research-results');
        researchElement.innerHTML = '';

        if (!research || research.length === 0) {
            researchElement.innerHTML = '<p style="color: #718096; font-style: italic;">No research data available</p>';
            return;
        }

        // Check if this is enhanced research with topic and claim separation
        const hasEnhancedResearch = research.some(item => 
            item.research_type && ['topic_background', 'claim_verification', 'enhanced_research_summary'].includes(item.research_type)
        );

        if (hasEnhancedResearch) {
            displayEnhancedResearchResults(research);
        } else {
            displayLegacyResearchResults(research);
        }
    }

    function displayEnhancedResearchResults(research) {
        const researchElement = document.getElementById('research-results');
        
        // Separate different types of research
        const summaryResults = research.filter(item => item.research_type === 'enhanced_research_summary');
        const topicResults = research.filter(item => item.research_type === 'topic_background');
        const claimResults = research.filter(item => item.research_type === 'claim_verification' || item.research_type === 'claim_verification_ai_only');
        
        // Display research summary first
        if (summaryResults.length > 0) {
            displayResearchSummary(summaryResults[0]);
        }
        
        // Display topic research
        if (topicResults.length > 0) {
            displayTopicResearch(topicResults);
        }
        
        // Display claim verification
        if (claimResults.length > 0) {
            displayClaimVerification(claimResults);
        }
    }

    function displayLegacyResearchResults(research) {
        const researchElement = document.getElementById('research-results');

        research.forEach((item, index) => {
            const researchDiv = document.createElement('div');
            researchDiv.className = 'research-item';
            researchDiv.style.marginBottom = '1.5rem';
            researchDiv.style.padding = '1.5rem';
            researchDiv.style.backgroundColor = '#ffffff';
            researchDiv.style.borderRadius = '12px';
            researchDiv.style.border = '1px solid #e2e8f0';
            researchDiv.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';

            // Determine colors based on verification status
            const statusColors = {
                'VERIFIED': { border: '#48bb78', bg: '#f0fff4', icon: '✅' },
                'PARTIALLY_VERIFIED': { border: '#ed8936', bg: '#fffaf0', icon: '⚠️' },
                'DISPUTED': { border: '#f56565', bg: '#fffbfb', icon: '❌' },
                'UNVERIFIABLE': { border: '#a0aec0', bg: '#f7fafc', icon: '❓' },
                'FALSE': { border: '#e53e3e', bg: '#fffbfb', icon: '❌' }
            };

            const statusInfo = statusColors[item.verification_status] || statusColors['UNVERIFIABLE'];
            researchDiv.style.borderLeft = `5px solid ${statusInfo.border}`;
            researchDiv.style.backgroundColor = statusInfo.bg;

            // Create claim header
            const claimText = typeof item.claim === 'string' ? item.claim : JSON.stringify(item.claim);
            const claimPreview = claimText.length > 120 ? claimText.substring(0, 120) + '...' : claimText;

            researchDiv.innerHTML = `
                <div style="margin-bottom: 1rem;">
                    <h4 style="color: #2d3748; margin-bottom: 0.5rem; font-weight: 600;">
                        Claim ${index + 1}: ${claimPreview}
                    </h4>
                </div>

                <!-- Status and Score Section -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; 
                           padding: 0.75rem; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: ${statusInfo.border}; margin-bottom: 0.25rem;">
                            ${item.status_message || `${statusInfo.icon} ${item.verification_status}`}
                        </div>
                        <div style="font-size: 0.875rem; color: #718096;">
                            Research Method: ${item.research_method || 'Standard verification'}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: ${statusInfo.border};">
                            ${item.truthfulness_score || 0}/100
                        </div>
                        <div style="font-size: 0.75rem; color: #718096;">
                            Truthfulness Score
                        </div>
                    </div>
                </div>

                <!-- Research Summary -->
                <div style="margin-bottom: 1rem; padding: 0.75rem; background: white; border-radius: 8px;">
                    <h5 style="color: #4a5568; margin-bottom: 0.5rem; font-weight: 600;">Research Summary</h5>
                    <p style="color: #2d3748; margin: 0; line-height: 1.5;">
                        ${item.research_summary || 'No summary available'}
                    </p>
                </div>

                <!-- Evidence Section -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div style="padding: 0.75rem; background: white; border-radius: 8px;">
                        <h5 style="color: #38a169; margin-bottom: 0.5rem; font-weight: 600;">
                            📈 Supporting Evidence
                        </h5>
                        <ul style="margin: 0; padding-left: 1rem; color: #2d3748;">
                            ${item.supporting_evidence && item.supporting_evidence.length > 0 
                                ? item.supporting_evidence.map(evidence => `<li style="margin-bottom: 0.25rem;">${evidence}</li>`).join('')
                                : '<li style="color: #718096; font-style: italic;">No supporting evidence found</li>'
                            }
                        </ul>
                    </div>
                    <div style="padding: 0.75rem; background: white; border-radius: 8px;">
                        <h5 style="color: #e53e3e; margin-bottom: 0.5rem; font-weight: 600;">
                            📉 Contradicting Evidence
                        </h5>
                        <ul style="margin: 0; padding-left: 1rem; color: #2d3748;">
                            ${item.contradicting_evidence && item.contradicting_evidence.length > 0 
                                ? item.contradicting_evidence.map(evidence => `<li style="margin-bottom: 0.25rem;">${evidence}</li>`).join('')
                                : '<li style="color: #718096; font-style: italic;">No contradicting evidence found</li>'
                            }
                        </ul>
                    </div>
                </div>

                <!-- Detailed Analysis -->
                <div style="margin-bottom: 1rem; padding: 0.75rem; background: white; border-radius: 8px;">
                    <h5 style="color: #4a5568; margin-bottom: 0.5rem; font-weight: 600;">Verification Details</h5>
                    <p style="color: #2d3748; margin: 0; line-height: 1.5; font-size: 0.9rem;">
                        ${item.verification_notes || 'No detailed verification notes available'}
                    </p>
                </div>

                <!-- Bottom Info Bar -->
                <div style="display: flex; justify-content: space-between; align-items: center; 
                           padding: 0.75rem; background: #f8f9fa; border-radius: 8px; font-size: 0.875rem;">
                    <div style="display: flex; gap: 1rem;">
                        <span style="color: #4a5568;">
                            <strong>Evidence Quality:</strong> 
                            <span style="color: ${getEvidenceQualityColor(item.evidence_quality)}; font-weight: 600;">
                                ${item.evidence_quality || 'Unknown'}
                            </span>
                        </span>
                        <span style="color: #4a5568;">
                            <strong>Confidence:</strong> 
                            <span style="color: ${getConfidenceColor(item.confidence_level)}; font-weight: 600;">
                                ${item.confidence_level || 'Unknown'}
                            </span>
                        </span>
                    </div>
                    <div style="padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem;
                               background: ${getRecommendationColor(item.recommendation).bg}; 
                               color: ${getRecommendationColor(item.recommendation).text};">
                        ${getRecommendationText(item.recommendation)}
                    </div>
                </div>

                <!-- Reliability Factors -->
                ${item.reliability_factors && item.reliability_factors.length > 0 ? `
                <div style="margin-top: 1rem; padding: 0.75rem; background: white; border-radius: 8px;">
                    <h5 style="color: #4a5568; margin-bottom: 0.5rem; font-weight: 600;">🔍 Reliability Factors</h5>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        ${item.reliability_factors.map(factor => 
                            `<span style="background: #e2e8f0; color: #4a5568; padding: 0.25rem 0.5rem; 
                                         border-radius: 12px; font-size: 0.8rem;">${factor}</span>`
                        ).join('')}
                    </div>
                </div>
                ` : ''}
            `;

            researchElement.appendChild(researchDiv);
        });
    }

    // Helper functions for styling
    function getEvidenceQualityColor(quality) {
        const colors = {
            'STRONG': '#38a169',
            'MODERATE': '#ed8936', 
            'WEAK': '#e53e3e',
            'INSUFFICIENT': '#a0aec0'
        };
        return colors[quality] || '#a0aec0';
    }

    function getConfidenceColor(confidence) {
        const colors = {
            'HIGH': '#38a169',
            'MEDIUM': '#ed8936',
            'LOW': '#e53e3e',
            'UNCERTAIN': '#a0aec0'
        };
        return colors[confidence] || '#a0aec0';
    }

    function getRecommendationColor(recommendation) {
        const colors = {
            'ACCEPT': { bg: '#c6f6d5', text: '#22543d' },
            'ACCEPT_WITH_CAUTION': { bg: '#fed7af', text: '#7b341e' },
            'QUESTION': { bg: '#fef5e7', text: '#744210' },
            'REJECT': { bg: '#fed7d7', text: '#742a2a' }
        };
        return colors[recommendation] || { bg: '#e2e8f0', text: '#4a5568' };
    }

    function getRecommendationText(recommendation) {
        const texts = {
            'ACCEPT': '✅ ACCEPT',
            'ACCEPT_WITH_CAUTION': '⚠️ ACCEPT WITH CAUTION', 
            'QUESTION': '❓ QUESTION',
            'REJECT': '❌ REJECT'
        };
        return texts[recommendation] || '❓ UNCLEAR';
    }

    function formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    function showError(message) {
        document.getElementById('error-message').textContent = message;
        errorSection.style.display = 'block';
        hideProgress();
        errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    function hideError() {
        errorSection.style.display = 'none';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    // Simple Analysis Display Functions
    function displaySimpleKeyClaims(claims) {
        const container = document.querySelector('.claims-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!claims || claims.length === 0) {
            container.innerHTML = '<p style="color: #718096; font-style: italic; padding: 1rem;">No key claims identified</p>';
            return;
        }
        
        claims.forEach(claim => {
            const claimDiv = document.createElement('div');
            claimDiv.className = 'claim-item';
            
            const evidenceBadgeClass = claim.evidence_level ? claim.evidence_level.toLowerCase() : 'none';
            
            claimDiv.innerHTML = `
                <div class="claim-header">
                    <div class="claim-text">${claim.claim || claim}</div>
                    <div class="evidence-badge ${evidenceBadgeClass}">${claim.evidence_level || 'Unknown'}</div>
                </div>
                ${claim.assessment ? `<div class="claim-meta">${claim.assessment}</div>` : ''}
            `;
            
            container.appendChild(claimDiv);
        });
    }
    
    function displaySimpleRedFlags(redFlags) {
        const container = document.querySelector('.red-flags-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!redFlags || redFlags.length === 0) {
            container.innerHTML = '<p style="color: #718096; font-style: italic; padding: 1rem;">No red flags identified</p>';
            return;
        }
        
        redFlags.forEach(flag => {
            const flagDiv = document.createElement('div');
            flagDiv.className = 'red-flag-item';
            
            const severityBadgeClass = flag.severity ? flag.severity.toLowerCase() : 'low';
            
            flagDiv.innerHTML = `
                <div class="red-flag-header">
                    <div class="red-flag-description">${flag.description || flag}</div>
                    <div class="severity-badge ${severityBadgeClass}">${flag.severity || 'Low'}</div>
                </div>
                ${flag.type ? `<div class="red-flag-meta">Type: ${flag.type}</div>` : ''}
                ${flag.example ? `<div class="red-flag-meta">Example: "${flag.example}"</div>` : ''}
            `;
            
            container.appendChild(flagDiv);
        });
    }
    
    function displaySimpleAssessment(assessment) {
        const container = document.querySelector('.assessment-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!assessment || Object.keys(assessment).length === 0) {
            container.innerHTML = '<p style="color: #718096; font-style: italic; padding: 1rem;">No assessment data available</p>';
            return;
        }
        
        const assessmentDiv = document.createElement('div');
        assessmentDiv.className = 'assessment-item';
        
        assessmentDiv.innerHTML = `
            ${assessment.main_assessment ? `
                <div style="margin-bottom: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3182ce;">
                    <strong style="color: #2d3748;">Assessment:</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #4a5568; line-height: 1.6;">${assessment.main_assessment}</p>
                </div>
            ` : ''}
            
            ${assessment.reliability_rating ? `
                <div style="margin-bottom: 1rem;">
                    <strong style="color: #2d3748;">Reliability Rating:</strong> 
                    <span style="font-weight: 600; color: ${getSimpleReliabilityColor(assessment.reliability_rating)}">${assessment.reliability_rating}</span>
                </div>
            ` : ''}
            
            ${assessment.confidence_score !== undefined ? `
                <div style="margin-bottom: 1rem;">
                    <strong style="color: #2d3748;">Confidence Score:</strong> 
                    <span style="font-weight: 600;">${Math.round(assessment.confidence_score * 100)}%</span>
                </div>
            ` : ''}
            
            ${assessment.strengths && assessment.strengths.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <strong style="color: #2d3748;">Strengths:</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        ${assessment.strengths.map(strength => `<li style="margin-bottom: 0.25rem;">${strength}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${assessment.weaknesses && assessment.weaknesses.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <strong style="color: #2d3748;">Weaknesses:</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        ${assessment.weaknesses.map(weakness => `<li style="margin-bottom: 0.25rem;">${weakness}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
        
        container.appendChild(assessmentDiv);
    }
    
    function displaySimpleSummary(summary) {
        const container = document.querySelector('.summary-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (!summary || Object.keys(summary).length === 0) {
            container.innerHTML = '<p style="color: #718096; font-style: italic; padding: 1rem;">No summary data available</p>';
            return;
        }
        
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'summary-item';
        
        summaryDiv.innerHTML = `
            ${summary.credibility_assessment ? `
                <div style="margin-bottom: 1.5rem; padding: 1rem; background: #f0fff4; border-radius: 8px; border-left: 4px solid #48bb78;">
                    <strong style="color: #22543d;">Key Takeaways:</strong>
                    <p style="margin: 0.5rem 0 0 0; color: #2d3748; line-height: 1.6;">${summary.credibility_assessment}</p>
                </div>
            ` : ''}
            
            ${summary.main_topic ? `
                <div style="margin-bottom: 1rem;">
                    <strong style="color: #2d3748;">Context:</strong> 
                    <span>${summary.main_topic}</span>
                </div>
            ` : ''}
            
            ${summary.key_points && summary.key_points.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <strong style="color: #2d3748;">Notable Claims:</strong>
                    <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                        ${summary.key_points.map(point => {
                            // Handle both string and object claims
                            const claimText = typeof point === 'object' ? (point.claim || point.text || point.description || JSON.stringify(point)) : point;
                            return `<li style="margin-bottom: 0.25rem;">${claimText}</li>`;
                        }).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${summary.recommendation ? `
                <div style="margin-bottom: 1rem; padding: 0.75rem; background: #fef5e7; border-radius: 6px;">
                    <strong style="color: #744210;">Recommendation:</strong> 
                    <span style="color: #2d3748;">${summary.recommendation}</span>
                </div>
            ` : ''}
        `;
        
        container.appendChild(summaryDiv);
    }
    
    function getSimpleReliabilityColor(rating) {
        switch (rating?.toLowerCase()) {
            case 'high': return '#16a34a';
            case 'medium': return '#d97706';
            case 'low': return '#dc2626';
            default: return '#6b7280';
        }
    }

    function displayResearchSummary(summary) {
        const researchElement = document.getElementById('research-results');
        
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'research-summary';
        summaryDiv.style.cssText = `
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        `;
        
        const topicInfo = summary.topic_info || {};
        const researchScope = summary.research_scope || {};
        
        summaryDiv.innerHTML = `
            <h3 style="margin: 0 0 1rem 0; color: white; font-size: 1.25rem;">
                🔍 Enhanced Research Summary
            </h3>
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                    ${summary.summary || 'Enhanced research completed'}
                </p>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div>
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem; opacity: 0.8;">Research Coverage</h4>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        ${researchScope.topic_coverage ? '<span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">📊 Topic Analysis</span>' : ''}
                        ${researchScope.claim_verification ? '<span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">✅ Claim Verification</span>' : ''}
                        ${researchScope.background_context ? '<span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">🏛️ Background Context</span>' : ''}
                    </div>
                </div>
                <div>
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem; opacity: 0.8;">Research Statistics</h4>
                    <div style="font-size: 0.9rem; opacity: 0.9;">
                        Topic Research: ${summary.total_topic_research || 0} • Claim Verification: ${summary.total_claim_verification || 0}
                    </div>
                </div>
            </div>
        `;
        
        researchElement.appendChild(summaryDiv);
    }
    
    function displayTopicResearch(topicResults) {
        const researchElement = document.getElementById('research-results');
        
        topicResults.forEach((topic, index) => {
            const topicDiv = document.createElement('div');
            topicDiv.className = 'topic-research';
            topicDiv.style.cssText = `
                margin-bottom: 2rem;
                padding: 1.5rem;
                background: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 5px solid #3182ce;
            `;
            
            const topicInfo = topic.topic_info || {};
            
            topicDiv.innerHTML = `
                <h3 style="color: #2d3748; margin: 0 0 1rem 0; font-size: 1.2rem;">
                    🏛️ Topic Background: ${topicInfo.main_topic || 'Topic Research'}
                </h3>
                
                ${topic.background_summary ? `
                <div style="margin-bottom: 1.5rem; padding: 1rem; background: #f7fafc; border-radius: 8px;">
                    <h4 style="color: #4a5568; margin: 0 0 0.5rem 0; font-size: 1rem;">Background Summary</h4>
                    <p style="color: #2d3748; margin: 0; line-height: 1.6;">
                        ${topic.background_summary}
                    </p>
                </div>
                ` : ''}
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                    ${topic.key_context && topic.key_context.length > 0 ? `
                    <div style="padding: 1rem; background: #f0fff4; border-radius: 8px; border-left: 3px solid #48bb78;">
                        <h4 style="color: #22543d; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Key Context</h4>
                        <ul style="margin: 0; padding-left: 1rem; color: #2d3748;">
                            ${topic.key_context.map(context => `<li style="margin-bottom: 0.25rem; font-size: 0.9rem;">${context}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    
                    ${topic.related_topics && topic.related_topics.length > 0 ? `
                    <div style="padding: 1rem; background: #fffaf0; border-radius: 8px; border-left: 3px solid #ed8936;">
                        <h4 style="color: #7b341e; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Related Topics</h4>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            ${topic.related_topics.map(relatedTopic => 
                                `<span style="background: #fed7af; color: #7b341e; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">${relatedTopic}</span>`
                            ).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    ${topic.historical_context ? `
                    <div style="padding: 1rem; background: #fefce8; border-radius: 8px;">
                        <h4 style="color: #744210; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Historical Context</h4>
                        <p style="color: #2d3748; margin: 0; font-size: 0.9rem; line-height: 1.5;">
                            ${topic.historical_context}
                        </p>
                    </div>
                    ` : ''}
                    
                    ${topic.current_status ? `
                    <div style="padding: 1rem; background: #f0f9ff; border-radius: 8px;">
                        <h4 style="color: #1e40af; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Current Status</h4>
                        <p style="color: #2d3748; margin: 0; font-size: 0.9rem; line-height: 1.5;">
                            ${topic.current_status}
                        </p>
                    </div>
                    ` : ''}
                </div>
                
                ${topic.expert_sources && topic.expert_sources.length > 0 ? `
                <div style="margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <h4 style="color: #4a5568; margin: 0 0 0.5rem 0; font-size: 0.9rem;">Expert Sources</h4>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        ${topic.expert_sources.map(source => 
                            `<span style="background: #e2e8f0; color: #4a5568; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">
                                ${source.source} (${source.credibility})
                            </span>`
                        ).join('')}
                    </div>
                </div>
                ` : ''}
                
                <div style="margin-top: 1rem; padding: 0.75rem; background: #f8f9fa; border-radius: 8px; font-size: 0.8rem; color: #718096;">
                    Research Method: ${topic.research_method || 'Enhanced Topic Research'} • 
                    Sources: ${topic.total_sources || 0} web sources analyzed
                </div>
            `;
            
            researchElement.appendChild(topicDiv);
        });
    }
    
    function displayClaimVerification(claimResults) {
        const researchElement = document.getElementById('research-results');
        
        // Add section header
        const headerDiv = document.createElement('div');
        headerDiv.style.cssText = `
            margin-bottom: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 8px;
            text-align: center;
        `;
        headerDiv.innerHTML = `
            <h3 style="margin: 0; font-size: 1.1rem;">
                ✅ Claim Verification Results (${claimResults.length} claims analyzed)
            </h3>
        `;
        researchElement.appendChild(headerDiv);
        
        // Display each claim verification using the existing logic
        claimResults.forEach((item, index) => {
            const researchDiv = document.createElement('div');
            researchDiv.className = 'research-item claim-verification';
            researchDiv.style.marginBottom = '1.5rem';
            researchDiv.style.padding = '1.5rem';
            researchDiv.style.backgroundColor = '#ffffff';
            researchDiv.style.borderRadius = '12px';
            researchDiv.style.border = '1px solid #e2e8f0';
            researchDiv.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';

            // Determine colors based on verification status
            const statusColors = {
                'VERIFIED': { border: '#48bb78', bg: '#f0fff4', icon: '✅' },
                'PARTIALLY_VERIFIED': { border: '#ed8936', bg: '#fffaf0', icon: '⚠️' },
                'DISPUTED': { border: '#f56565', bg: '#fffbfb', icon: '❌' },
                'UNVERIFIABLE': { border: '#a0aec0', bg: '#f7fafc', icon: '❓' },
                'FALSE': { border: '#e53e3e', bg: '#fffbfb', icon: '❌' }
            };

            const statusInfo = statusColors[item.verification_status] || statusColors['UNVERIFIABLE'];
            researchDiv.style.borderLeft = `5px solid ${statusInfo.border}`;
            researchDiv.style.backgroundColor = statusInfo.bg;

            // Create claim header
            const claimText = typeof item.claim === 'string' ? item.claim : JSON.stringify(item.claim);
            const claimPreview = claimText.length > 120 ? claimText.substring(0, 120) + '...' : claimText;

            researchDiv.innerHTML = `
                <div style="margin-bottom: 1rem;">
                    <h4 style="color: #2d3748; margin-bottom: 0.5rem; font-weight: 600;">
                        Claim ${index + 1}: ${claimPreview}
                    </h4>
                </div>

                <!-- Status and Score Section -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; 
                           padding: 0.75rem; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; color: ${statusInfo.border}; margin-bottom: 0.25rem;">
                            ${item.status_message || `${statusInfo.icon} ${item.verification_status}`}
                        </div>
                        <div style="font-size: 0.875rem; color: #718096;">
                            Research Method: ${item.research_method || 'Standard verification'}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: ${statusInfo.border};">
                            ${item.truthfulness_score || 0}/100
                        </div>
                        <div style="font-size: 0.75rem; color: #718096;">
                            Truthfulness Score
                        </div>
                    </div>
                </div>

                <!-- Research Summary -->
                <div style="margin-bottom: 1rem; padding: 0.75rem; background: white; border-radius: 8px;">
                    <h5 style="color: #4a5568; margin-bottom: 0.5rem; font-weight: 600;">Research Summary</h5>
                    <p style="color: #2d3748; margin: 0; line-height: 1.5;">
                        ${item.research_summary || 'No summary available'}
                    </p>
                </div>

                <!-- Evidence Section -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                    <div style="padding: 0.75rem; background: white; border-radius: 8px;">
                        <h5 style="color: #38a169; margin-bottom: 0.5rem; font-weight: 600;">
                            📈 Supporting Evidence
                        </h5>
                        <ul style="margin: 0; padding-left: 1rem; color: #2d3748;">
                            ${item.supporting_evidence && item.supporting_evidence.length > 0 
                                ? item.supporting_evidence.map(evidence => `<li style="margin-bottom: 0.25rem;">${evidence}</li>`).join('')
                                : '<li style="color: #718096; font-style: italic;">No supporting evidence found</li>'
                            }
                        </ul>
                    </div>
                    <div style="padding: 0.75rem; background: white; border-radius: 8px;">
                        <h5 style="color: #e53e3e; margin-bottom: 0.5rem; font-weight: 600;">
                            📉 Contradicting Evidence
                        </h5>
                        <ul style="margin: 0; padding-left: 1rem; color: #2d3748;">
                            ${item.contradicting_evidence && item.contradicting_evidence.length > 0 
                                ? item.contradicting_evidence.map(evidence => `<li style="margin-bottom: 0.25rem;">${evidence}</li>`).join('')
                                : '<li style="color: #718096; font-style: italic;">No contradicting evidence found</li>'
                            }
                        </ul>
                    </div>
                </div>

                <!-- Bottom Info Bar -->
                <div style="display: flex; justify-content: space-between; align-items: center; 
                           padding: 0.75rem; background: #f8f9fa; border-radius: 8px; font-size: 0.875rem;">
                    <div style="display: flex; gap: 1rem;">
                        <span style="color: #4a5568;">
                            <strong>Evidence Quality:</strong> 
                            <span style="color: ${getEvidenceQualityColor(item.evidence_quality)}; font-weight: 600;">
                                ${item.evidence_quality || 'Unknown'}
                            </span>
                        </span>
                        <span style="color: #4a5568;">
                            <strong>Web Sources:</strong> 
                            <span style="font-weight: 600;">
                                ${item.web_sources_found || 0}
                            </span>
                        </span>
                    </div>
                    <div style="padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem;
                               background: ${getRecommendationColor(item.recommendation).bg}; 
                               color: ${getRecommendationColor(item.recommendation).text};">
                        ${getRecommendationText(item.recommendation)}
                    </div>
                </div>
            `;

            researchElement.appendChild(researchDiv);
        });
    }

    // Video Generation Functions
    let currentVideoRequestId = null;
    let videoProgressInterval = null;
    
    function showVideoGenerationSection(requestId) {
        currentVideoRequestId = requestId;
        const videoSection = document.getElementById('video-generation-section');
        if (videoSection) {
            videoSection.style.display = 'block';
            
            // Set up event listeners
            setupVideoEventListeners();
        }
    }
    
    function setupVideoEventListeners() {
        const generateBtn = document.getElementById('generate-video-btn');
        const generateAnotherBtn = document.getElementById('generate-another-btn');
        
        if (generateBtn && !generateBtn.hasVideoListener) {
            generateBtn.addEventListener('click', handleVideoGeneration);
            generateBtn.hasVideoListener = true;
        }
        
        if (generateAnotherBtn && !generateAnotherBtn.hasVideoListener) {
            generateAnotherBtn.addEventListener('click', resetVideoGeneration);
            generateAnotherBtn.hasVideoListener = true;
        }
    }
    
    async function handleVideoGeneration() {
        if (!currentVideoRequestId) {
            showVideoError('No analysis data available for video generation');
            return;
        }
        
        try {
            // Get selected options
            const videoType = document.querySelector('input[name="videoType"]:checked')?.value || 'social';
            const includeCredibility = document.getElementById('include-credibility')?.checked || true;
            const includeFactChecks = document.getElementById('include-fact-checks')?.checked || true;
            const includeSpeakers = document.getElementById('include-speakers')?.checked || true;
            
            // Set loading state
            setVideoLoadingState(true);
            showVideoProgress();
            
            // Prepare request
            const clipConfig = {
                type: videoType,
                target_duration: videoType === 'social' ? 25 : 120,
                style: 'motion_graphics',
                include_credibility: includeCredibility,
                include_fact_checks: includeFactChecks,
                include_speakers: includeSpeakers
            };
            
            console.log('Requesting video generation with config:', clipConfig);
            
            // Send request to TruthScore backend
            const response = await fetch('/generate_video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    request_id: currentVideoRequestId,
                    clip_config: clipConfig
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Video generation started:', result);
            
            // Start polling for progress
            if (result.clip_id) {
                startVideoProgressPolling(result.clip_id);
            }
            
        } catch (error) {
            console.error('Video generation error:', error);
            showVideoError(`Failed to start video generation: ${error.message}`);
            setVideoLoadingState(false);
            hideVideoProgress();
        }
    }
    
    function startVideoProgressPolling(clipId) {
        if (videoProgressInterval) {
            clearInterval(videoProgressInterval);
        }
        
        console.log('Starting video progress polling for clip:', clipId);
        
        // Poll every 2 seconds
        videoProgressInterval = setInterval(async () => {
            try {
                const response = await fetch(`http://localhost:9000/progress/${clipId}`);
                
                if (!response.ok) {
                    throw new Error(`Progress check failed: ${response.status}`);
                }
                
                const progress = await response.json();
                console.log('Video progress update:', progress);
                
                updateVideoProgress(progress);
                
                // Check if completed
                if (progress.status === 'completed') {
                    clearInterval(videoProgressInterval);
                    videoProgressInterval = null;
                    
                    // Get final status from TruthScore
                    const statusResponse = await fetch(`/video_status/${clipId}`);
                    if (statusResponse.ok) {
                        const finalStatus = await statusResponse.json();
                        showVideoResult(finalStatus);
                    } else {
                        showVideoResult(progress);
                    }
                    
                } else if (progress.status === 'failed') {
                    clearInterval(videoProgressInterval);
                    videoProgressInterval = null;
                    showVideoError(`Video generation failed: ${progress.error || 'Unknown error'}`);
                }
                
            } catch (error) {
                console.error('Error checking video progress:', error);
                // Don't clear interval on network errors, keep trying
            }
        }, 2000);
    }
    
    function updateVideoProgress(progress) {
        // Update progress bar
        const progressFill = document.getElementById('video-progress-fill');
        const progressPercentage = document.getElementById('video-progress-percentage');
        const progressMessage = document.getElementById('video-progress-message');
        
        if (progressFill) {
            progressFill.style.width = `${progress.progress || 0}%`;
        }
        
        if (progressPercentage) {
            progressPercentage.textContent = `${progress.progress || 0}%`;
        }
        
        if (progressMessage) {
            progressMessage.textContent = progress.current_step || 'Processing...';
        }
        
        // Update step indicators
        if (progress.steps) {
            progress.steps.forEach((step, index) => {
                const stepElement = document.getElementById(`video-step-${['analysis', 'tts', 'assembly', 'render'][index]}`);
                if (stepElement) {
                    if (step.completed) {
                        stepElement.classList.add('completed');
                    } else {
                        stepElement.classList.remove('completed');
                    }
                }
            });
        }
    }
    
    function showVideoResult(result) {
        setVideoLoadingState(false);
        hideVideoProgress();
        
        const resultSection = document.getElementById('video-result-section');
        if (resultSection) {
            resultSection.style.display = 'block';
            
            // Update metadata
            const metadata = result.metadata || {};
            const durationEl = document.getElementById('video-duration');
            const sizeEl = document.getElementById('video-size');
            
            if (durationEl) {
                durationEl.textContent = `Duration: ${metadata.duration || '--'}`;
            }
            
            if (sizeEl) {
                sizeEl.textContent = `Size: ${metadata.file_size || '--'}`;
            }
            
            // Set up download link
            const downloadLink = document.getElementById('download-video-link');
            if (downloadLink && result.download_url) {
                downloadLink.href = result.download_url;
                downloadLink.style.display = 'inline-block';
            }
        }
        
        // Scroll to result
        if (resultSection) {
            resultSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    function showVideoProgress() {
        const progressSection = document.getElementById('video-progress-section');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
    }
    
    function hideVideoProgress() {
        const progressSection = document.getElementById('video-progress-section');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
    }
    
    function setVideoLoadingState(loading) {
        const generateBtn = document.getElementById('generate-video-btn');
        const btnText = generateBtn?.querySelector('.btn-text');
        const btnLoader = generateBtn?.querySelector('.btn-loading');
        
        if (generateBtn) {
            generateBtn.disabled = loading;
        }
        
        if (btnText) {
            btnText.style.display = loading ? 'none' : 'inline';
        }
        
        if (btnLoader) {
            btnLoader.style.display = loading ? 'inline' : 'none';
        }
    }
    
    function showVideoError(message) {
        // Create or update error display
        let errorDiv = document.getElementById('video-error-message');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.id = 'video-error-message';
            errorDiv.className = 'video-error';
            
            const videoSection = document.getElementById('video-generation-section');
            if (videoSection) {
                videoSection.appendChild(errorDiv);
            }
        }
        
        errorDiv.innerHTML = `
            <div class="error-content">
                <span class="error-icon">⚠️</span>
                <span class="error-text">${message}</span>
                <button type="button" onclick="hideVideoError()" class="error-close">×</button>
            </div>
        `;
        
        errorDiv.style.display = 'block';
    }
    
    function hideVideoError() {
        const errorDiv = document.getElementById('video-error-message');
        if (errorDiv) {
            errorDiv.style.display = 'none';
        }
    }
    
    function resetVideoGeneration() {
        // Reset UI state for another generation
        hideVideoProgress();
        const resultSection = document.getElementById('video-result-section');
        if (resultSection) {
            resultSection.style.display = 'none';
        }
        
        hideVideoError();
        setVideoLoadingState(false);
        
        // Clear any active polling
        if (videoProgressInterval) {
            clearInterval(videoProgressInterval);
            videoProgressInterval = null;
        }
    }
    
    // Make video functions globally accessible
    window.hideVideoError = hideVideoError;
    window.showVideoGenerationSection = showVideoGenerationSection;
    window.handleVideoGeneration = handleVideoGeneration;
    window.resetVideoGeneration = resetVideoGeneration;

    // Global function for retry button
    window.hideError = hideError;
}); 