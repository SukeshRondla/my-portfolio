import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
// import profilePhoto from '/Users/sukeshreddy/Downloads/MyWebsite/my-portfolio/src/assets/Photoroom_20250123_221608.jpg'
// import pathvisvrImg from '/Users/sukeshreddy/Downloads/MyWebsite/my-portfolio/src/assets/pathvis.jpg'
// import retinalImg from '/Users/sukeshreddy/Downloads/MyWebsite/my-portfolio/src/assets/retinal.jpg'
// import nlpImg from '/Users/sukeshreddy/Downloads/MyWebsite/my-portfolio/src/assets/nlp.jpg'
// import semanticSearchImg from '/Users/sukeshreddy/Downloads/MyWebsite/my-portfolio/src/assets/semantic-search.jpg'
import './App.css'
import { insightsData } from './insightsData';
import InsightDetail from './InsightDetail';

const workData = [
  {
    title: 'Machine Learning Engineer',
    company: 'Alchemy AI - Cognitus',
    companyUrl: 'https://alchemyai.com/',
    dates: 'June 2024 – Present',
    bullets: [
      "Led customer-facing demos and onboarding sessions for Alchemy's AI agents, translating client use cases into deployable automation workflows and iterating in real-time to address business-specific challenges.",
      "Collaborated directly with enterprise clients to prototype and deploy intelligent agents tailored to their operational workflows, accelerating time-to-value and driving adoption through personalized solutioning.",
      "Integrated AI pipelines with external systems (e.g., REST APIs, webhooks, third-party data platforms) to enable end-to-end automation, reducing manual intervention and improving throughput.",
      "Tuned prompt logic and fallback flows across agents to optimize performance, ensuring high accuracy and resilience in production scenarios involving variable inputs and user behavior.",
      "Translated customer needs into technical specs, working directly with product and customer teams to define automation rules, business logic, and exception handling in intelligent agents.",
      "Deployed and versioned agents in containerized environments using Docker and GitHub Actions, enabling CI/CD workflows and ensuring reproducible, scalable deployments.",
      "Monitored agent telemetry, accuracy, and edge cases, conducting A/B testing and leveraging observability tooling to trigger iterative improvements and model fine-tuning.",
      "Served as a trusted technical advisor, consulting clients and internal stakeholders on scalable architecture, prompt design, and long-term maintainability of AI-driven systems.",
      "Mentored junior engineers on applied AI practices, agent deployment patterns, and customer-first iteration using Agile and sprint-based delivery."
    ]
  },
  {
    title: 'Data Scientist',
    company: 'Retinawise AI – RCN Networks',
    companyUrl: '',
    dates: 'June 2023 – March 2024',
    bullets: [
      "Fine-tuned multimodal AI models using PyTorch and Hugging Face Transformers for retinal disease classification, improving diagnostic suggestion alignment by 22%.",
      "Collaborated with ophthalmologists to define structured outputs for differential diagnoses, enhancing the clinical relevance of generated summaries.",
      "Developed scalable preprocessing and evaluation pipelines for grayscale retinal image data using Pandas, NumPy, and OpenCV.",
      "Deployed inference services using FastAPI and Docker, orchestrated on Azure Kubernetes Service (AKS) with auto-scaling for batch predictions.",
      "Integrated Azure Blob Storage for dataset management and leveraged Azure Key Vault to securely handle authentication tokens.",
      "Built interactive Streamlit tools to visualize predictions and assist internal review of model outputs by medical analysts.",
      "Evaluated vision-language models such as BLIP, BioViL, and CLIP for structured ophthalmic diagnosis, focusing on image-text alignment and disease pattern recognition.",
      "Developed prompt-based inference workflows to generate explainable diagnostic summaries from fundus scans, leveraging pretrained image encoders and clinical language models."
    ]
  },
  {
    title: 'Machine Learning Engineer',
    company: 'Vamstar',
    companyUrl: 'https://vamstar.io/',
    dates: 'Feb 2020 – Dec 2021',
    bullets: [
      "Built and productionized NLP pipelines using spaCy, Scikit-learn, and Hugging Face Transformers to extract entities and classify healthcare procurement tenders, improving labeling efficiency by 70%.",
      "Developed multilingual classifiers with BERT-base-multilingual-cased, achieving 85%+ F1-score on tender documents across English, German, and French.",
      "Implemented UNSPSC classification using LightGBM and rule-based feature engineering, increasing product mapping accuracy by 20%.",
      "Created entity resolution pipelines using RapidFuzz, TF-IDF, and MinHash LSH, deduplicating supplier/buyer records with over 90% precision.",
      "Engineered a semantic search and recommendation engine using SBERT and FAISS, boosting supplier match coverage by 35%.",
      "Deployed ML services via Docker and Azure Kubernetes Service (AKS), orchestrated with Azure ML pipelines for scalable inference.",
      "Built a harmonized procurement data lake using Azure Data Lake Gen2, improving model reliability and team-wide data access.",
      "Integrated inference APIs into the product using FastAPI and Celery, enabling real-time supplier recommendations.",
      "Partnered with product and data teams to define standardized taxonomies across 20+ countries, ensuring consistent downstream analytics."
    ]
  },
  {
    title: 'Research and Development Intern',
    company: 'IIIT Hyderabad',
    companyUrl: 'https://www.iiit.ac.in/',
    dates: 'May 2018 - Nov 2018',
    bullets: [
      "Cleaned and preprocessed raw datasets using Pandas and NumPy, handling missing values, encoding categorical variables, and scaling numerical features to prepare data for ML model training.",
      "Performed exploratory data analysis (EDA) using Matplotlib and Seaborn to identify outliers, visualize feature distributions, and guide data-driven feature selection."
    ]
  }
];

// Example data for Featured Projects & Publications - REPLACE WITH YOUR ACTUAL PROJECTS AND PUBLICATIONS
const projectsData = [
  {
    imgSrc: "pathVis.png",
    altText: 'PathVisVR project',
    title: 'PathVisVR: Immersive Medical Image Viewer',
    authors: 'Sukesh Reddy Rondla, Alafia AI Team', // Example Authors - Replace
    venue: 'CES 2025', // Example Venue - Replace
    description: 'Engineered a novel immersive VR application for visualizing gigapixel digital pathology slides and integrating multi-omics datasets. Utilized <b>PyTorch</b> for image analysis backend, developed interactive features using a modern VR framework, and designed scalable data loading pipelines leveraging <b>Pandas</b> and <b>NumPy</b>. Presented the system\'s capabilities and clinical utility at <b>CES 2025</b>.', // Example Description - Replace
    links: [
      { text: 'Project', url: '#' }, // Replace # with actual link
      { text: 'Code', url: '#' }
    ]
  },
  {
    imgSrc: "retinaWise.png",
    altText: 'Retinal Disease AI Diagnosis',
    title: 'Retinal Disease AI Diagnosis',
    authors: 'Sukesh Reddy Rondla, Retinawise AI Team', // Example Authors - Replace
    venue: 'Retinawise AI, 2023', // Example Venue - Replace
    description: 'Fine-tuned multimodal AI models, including variants of <b>BLIP</b> and <b>CLIP</b> (<b>Vision Models</b>) using <b>PyTorch</b> and <b>Hugging Face Transformers</b>, for automated retinal disease classification from grayscale fundus scans. Developed scalable preprocessing pipelines with <b>OpenCV</b> and <b>Pandas</b>. Deployed the inference service via <b>FastAPI</b> and <b>Docker</b> on <b>Azure Kubernetes Service (AKS)</b> with auto-scaling, integrating with <b>Azure Blob Storage</b> and <b>Azure Key Vault</b>. Achieved a <b>22% improvement</b> in diagnostic suggestion alignment through iterative fine-tuning and collaboration with ophthalmologists.', // Example Description - Replace
    links: [
      { text: 'Project', url: '#' }, // Replace # with actual link
      { text: 'Code', url: '#' }
    ]
  },
  {
    imgSrc: "Healthcare.png",
    altText: 'Healthcare Procurement Tender NLP Classifier',
    title: 'Healthcare Procurement Tender NLP Classifier',
    authors: 'Sukesh Reddy Rondla, Vamstar Team', // Example Authors - Replace
    venue: 'Vamstar, Feb 2020 – Dec 2021', // Example Venue - Replace
    description: 'Built and productionized an end-to-end NLP pipeline using <b>spaCy</b>, <b>Scikit-learn</b>, and <b>BERT-base-multilingual-cased</b> (<b>Hugging Face Transformers</b>) to extract entities and classify healthcare procurement tenders across multiple languages. Implemented <b>LightGBM</b> for UNSPSC classification and developed entity resolution using <b>RapidFuzz</b> and <b>TF-IDF</b>. Deployed the model as a <b>FastAPI</b> service via <b>Docker</b> and <b>Azure Kubernetes Service (AKS)</b>, integrated with <b>Azure Data Lake Gen2</b> for data management. This system improved labeling efficiency by <b>70%</b>.', // Example Description - Replace
    links: [
      { text: 'Project', url: 'https://github.com/SukeshRondla/Healthcare-Procurement-Tender-NLP-Classifier/tree/main' }, // Replace # with actual link
      { text: 'Code', url: '#' }
    ]
  },
  {
    imgSrc: "semantic.png",
    altText: 'Semantic Search and Recommendation Engine',
    title: 'Semantic Search and Recommendation Engine',
    authors: 'Sukesh Reddy Rondla, Vamstar Team', // Example Authors - Replace
    venue: 'Vamstar, Feb 2020 – Dec 2021', // Example Venue - Replace
    description: 'Engineered a semantic search and recommendation engine for supplier discovery using <b>SBERT</b> and <b>FAISS</b>. Developed scalable data processing with <b>PySpark</b> and implemented real-time inference APIs with <b>FastAPI</b> and <b>Celery</b>. This system boosted supplier match coverage by <b>35%</b>, demonstrating effective application of advanced NLP techniques for business impact.', // Example Description - Replace
    links: [
      { text: 'Project', url: '#' }, // Replace # with actual link
      { text: 'Code', url: '#' }
    ]
  }
];

function App() {
  const [openJob, setOpenJob] = useState(null);

  return (
    <Router basename='/'>
      <Routes>
        <Route path="/insight/:slug" element={<InsightDetail />} />
        <Route path="/" element={(
          <div className="main-content-area">
            {/* Header Section */}
            <header className="header-flex">
              <div className="header-left">
                <h1 className="main-title">Sukesh Reddy Rondla</h1>
                <h2 className="subtitle">Machine Learning Engineer | AI, NLP, Computer Vision, MLOps</h2>
                <p className="summary">
                  I'm a Machine Learning Engineer focused on building impactful AI systems across healthcare, automation, and multimodal applications. My work includes training NLP and vision models with PyTorch and Hugging Face, deploying scalable pipelines with FastAPI and Docker, and improving model trust and interpretability in clinical settings. I specialize in taking models from research to production across AWS, Azure, and GCP.
                </p>
              </div>
              <div className="header-right">
                <img src="/profile-photo.jpg" alt="Sukesh Reddy Rondla" className="profile-photo-large" />
                <div className="contact-row" style={{ marginTop: '1.2rem', textAlign: 'center' }}>
                  <a href="mailto:Contact.sukeshreddyrondla@gmail.com" title="Email">📧</a>
                  <a href="https://www.linkedin.com/in/sukeshreddyrondla/" target="_blank" rel="noopener noreferrer" title="LinkedIn">in</a>
                  <a href="https://www.SukeshReddyRondla.com" target="_blank" rel="noopener noreferrer" title="Website">🌐</a>
                  <a href="https://github.com/SukeshRondla" target="_blank" rel="noopener noreferrer" title="GitHub">GitHub</a>
                </div>
                <div className="contact-details" style={{ textAlign: 'center' }}>
                  <span>Contact.sukeshreddyrondla@gmail.com</span> | <span>+1 (970)-827-8470</span>
                </div>
                <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                  <a href="/SukeshRondla.pdf" download className="download-button-new">
                    <div className="button-outer">
                      <div className="button-inner">
                        <span>Download Resume</span>
                      </div>
                    </div>
                  </a>
                </div>
              </div>
            </header>
            
            {/* News Section */}
            <section className="news-section">
              <h3>Recent News</h3>
              <div className="news-list">
                <div className="news-item">
                  <div className="news-date">May 2025</div>
                  <div className="news-desc"><b>Showcased Alchemy AI at SAP Sapphire 2025</b><br/>
                    <b>Miami, FL</b> — Presented <b>Alchemy AI</b>, demonstrating enterprise-grade AI agents for SAP environments. Showcased integrations across supply chain workflows, real-time analytics, and automation built using <b>Alchemy's no-code platform</b>.
                  </div>
                </div>
                <div className="news-item">
                  <div className="news-date">Nov 2024</div>
                  <div className="news-desc"><b>Google Developer Groups</b> — <b>San Jose, CA</b><br/>Designed AI-powered tools for smart vertical communities, including an <b>AI Concierge</b> and <b>ESG-focused facility management solutions</b> using <b>Google Cloud AI</b> and <b>Aparavi</b>.</div>
                </div>
                <div className="news-item">
                  <div className="news-date">Oct 2024</div>
                  <div className="news-desc"><b>Apple VisionOS Developer Meet</b> — <b>San Jose, CA</b><br/>Explored multimodal app development using <b>Apple VisionOS</b>. Attended sessions focused on building immersive <b>augmented reality experiences</b> with <b>Swift</b>, <b>RealityKit</b>, and <b>Apple's spatial computing SDKs</b>.</div>
                </div>
                <div className="news-item">
                  <div className="news-date">June 2024</div>
                  <div className="news-desc">Presented <b>PathVisVR</b> with <b>Alafia AI</b>, an innovative <b>digital pathology</b> and <b>medical imaging platform</b>. Highlighted real-time <b>3D scan visualization</b>, <b>deep learning diagnostics</b>, and <b>clinician-friendly workflows</b>.</div>
                </div>
                <div className="news-item">
                  <div className="news-date">May 2024</div>
                  <div className="news-desc">Joined <b>Alchemy AI (Cognitus)</b> — <b>Remote / Dallas, TX</b><br/>Started as <b>Machine Learning Engineer</b>, leading enterprise <b>AI agent deployments</b>. Built <b>LLM-powered assistants</b> for <b>ERP systems</b>, enabling dynamic <b>workflow automation</b> and <b>domain-specific language understanding</b>.</div>
                </div>
                <div className="news-item">
                  <div className="news-date">March 2024</div>
                  <div className="news-desc">Fine-tuned multimodal AI models for <b>retinal disease classification</b>. Improved diagnostic accuracy by <b>22%</b> using <b>transfer learning</b> on grayscale fundus images and <b>vision transformer architectures</b>.</div>
                </div>
                <div className="news-item">
                  <div className="news-date">Jan 10, 2022</div>
                  <div className="news-desc"><b>Western Illinois University</b> — <b>Macomb, IL</b><br/>Began <b>MS in Computer Science</b>, focusing on <b>machine learning</b>, <b>data systems</b>, and <b>computer vision</b>. Built early research prototypes in <b>semantic image retrieval</b> and <b>document classification</b>.</div>
                </div>
              </div>
            </section>

            {/* Work Experience Section */}
            <section className="work-section">
              <h3>Work Experience</h3>
              {workData.map((job, idx) => (
                <div
                  className={`work-job work-accordion${openJob === idx ? ' open' : ''}`}
                  key={job.title + job.company}
                >
                  <div
                    className="work-summary"
                    onClick={() => setOpenJob(openJob === idx ? null : idx)}
                    style={{ cursor: 'pointer' }}
                    tabIndex="-1"
                  >
                    <div className="work-title">{job.title}</div>
                      <div className="work-company">{job.company}</div>
                    <div className="work-dates">{job.dates}</div>
                  </div>
                  {openJob === idx && (
                    <ul className="work-list">
                      {job.bullets.map((b, i) => (
                        <li key={i} dangerouslySetInnerHTML={{ __html: b }}></li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </section>

            {/* Projects Section */}
            <section className="projects-section">
              <h3>Featured Projects & Publications</h3>
              <div className="projects-grid"> {/* Use a grid for projects */}
                {projectsData.map((project, idx) => (
                  <div className="pub-card" key={idx}>
                    {project.imgSrc && <img src={project.imgSrc} alt={project.altText} className="pub-thumb" />} {/* Use imgSrc */}
                    {!project.imgSrc && <div className="pub-thumb placeholder">No Image</div>} {/* Placeholder if no image */}
                    <div className="pub-info">
                      <div className="pub-title">{project.title}</div>
                      <div className="pub-authors">{project.authors}</div>
                      <div className="pub-venue">{project.venue}</div>
                      <div className="pub-description" dangerouslySetInnerHTML={{ __html: project.description }}></div> {/* Render HTML */} 
                      <div className="pub-links">
                        {project.links.map((link, linkIdx) => (
                          <a key={linkIdx} href={link.url} target="_blank" rel="noopener noreferrer">{link.text}</a>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* Technical Insights Section */}
            <section className="technical-insights-section">
              <h3>Technical Insights</h3>
              <div className="insights-list"> {/* Keep list for insights */}
                {insightsData.map((insight, idx) => (
                  <div className="insight-item" key={idx}>
                    <div className="insight-date">{insight.date}</div>
                    <div className="insight-title">
                      <Link to={`/insight/${insight.slug}`}>{insight.title}</Link>
                    </div>
                    <div className="insight-summary">{insight.summary}</div>
                  </div>
                ))}
              </div>
            </section>

            {/* Footer (Optional) */}
            {/* Add a footer here if desired */}

          </div>
        )} />
      </Routes>
    </Router>
  );
}

export default App;
