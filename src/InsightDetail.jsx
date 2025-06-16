import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { insightsData } from './insightsData';
import './InsightDetail.css'; // Import specific styles for this component

function InsightDetail() {
  const { slug } = useParams();
  const insight = insightsData.find(item => item.slug === slug);

  if (!insight) {
    return <div>Insight not found</div>;
  }

  return (
    <div className="main-wrapper">
      <header className="header-flex">
        <div className="header-left">
          {/* You might want to add some basic header/nav here */}
        </div>
      </header>
      
      <section className="insight-detail-section">
        <h2>{insight.title}</h2>
        <div className="insight-date">{insight.date}</div>
        <div className="insight-content" dangerouslySetInnerHTML={{ __html: insight.content }}></div>
        {insight.link && insight.link !== '#' && (
          <div className="insight-external-link">
            <a href={insight.link} target="_blank" rel="noopener noreferrer">Read original post</a>
          </div>
        )}
        <div className="back-link" style={{ marginTop: '20px' }}>
          <Link to="/">Back to Home</Link>
        </div>
      </section>
    </div>
  );
}

export default InsightDetail; 