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
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
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
                        throw new Error(resultsData.error || 'Analysis failed');
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
                                showError(resultsData.error || 'Analysis failed');
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
                            showError(resultsData.error || 'Analysis failed');
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
        // Hide error and progress, show results
        hideError();
        hideProgress();
        resultsSection.style.display = 'block';
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Display content info
        document.getElementById('content-title').textContent = data.title || 'Unknown Title';
        document.getElementById('analysis-timestamp').textContent = `Analyzed on: ${formatTimestamp(data.timestamp)}`;

        // Display credibility score
        displayCredibilityScore(data.analysis.credibility_score);

        // Display analysis details
        displayKeyClaimsList('key-claims', data.analysis.key_claims);
        displayKeyClaimsList('red-flags', data.analysis.red_flags);
        document.getElementById('factual-accuracy').textContent = data.analysis.factual_accuracy;
        displayKeyClaimsList('bias-indicators', data.analysis.bias_indicators);
        document.getElementById('evidence-quality').textContent = data.analysis.evidence_quality;
        document.getElementById('analysis-summary').textContent = data.analysis.analysis_summary;
        
        // Display chain of thought reasoning
        const chainOfThought = data.analysis.chain_of_thought || 'Chain of thought reasoning not available for this analysis.';
        document.getElementById('chain-of-thought').textContent = chainOfThought;

        // Display transcript
        if (data.transcript && typeof data.transcript === 'object') {
            document.getElementById('transcript-content').textContent = data.transcript.full_text || 'No transcript available';
        } else {
            document.getElementById('transcript-content').textContent = data.transcript || 'No transcript available';
        }

        // Display research results
        displayResearchResults(data.research);
    }

    function displayCredibilityScore(score) {
        const scoreElement = document.getElementById('score-value');
        const scoreDisplay = document.querySelector('.score-display');
        const scoreDescription = document.getElementById('score-description');

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

    // Global function for retry button
    window.hideError = hideError;
}); 