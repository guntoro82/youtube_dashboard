import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizations:
    """Create interactive visualizations for YouTube analytics"""
    
    def __init__(self):
        """Initialize visualization settings"""
        self.color_palette = px.colors.qualitative.Set3
        self.primary_color = '#1f77b4'
    
    def create_views_timeline(self, df):
        """
        Create timeline chart of video views
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Timeline chart
        """
        fig = px.scatter(
            df.sort_values('published_at'),
            x='published_at',
            y='view_count',
            size='engagement_rate',
            color='engagement_rate',
            hover_data=['title', 'like_count', 'comment_count'],
            title="Video Performance Over Time",
            labels={
                'published_at': 'Upload Date',
                'view_count': 'Views',
                'engagement_rate': 'Engagement Rate'
            }
        )
        
        fig.update_layout(
            xaxis_title="Upload Date",
            yaxis_title="Views",
            hovermode='closest'
        )
        
        return fig
    
    def create_engagement_distribution(self, df):
        """
        Create engagement rate distribution histogram
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Histogram
        """
        fig = px.histogram(
            df,
            x='engagement_rate',
            nbins=20,
            title="Engagement Rate Distribution",
            labels={'engagement_rate': 'Engagement Rate', 'count': 'Number of Videos'}
        )
        
        # Add mean line
        mean_engagement = df['engagement_rate'].mean()
        fig.add_vline(
            x=mean_engagement,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_engagement:.3f}"
        )
        
        fig.update_layout(
            xaxis_title="Engagement Rate",
            yaxis_title="Number of Videos"
        )
        
        return fig
    
    def create_engagement_vs_views(self, df):
        """
        Create scatter plot of engagement rate vs views
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot
        """
        fig = px.scatter(
            df,
            x='view_count',
            y='engagement_rate',
            color='like_count',
            size='comment_count',
            hover_data=['title'],
            title="Engagement Rate vs Views",
            labels={
                'view_count': 'Views',
                'engagement_rate': 'Engagement Rate',
                'like_count': 'Likes',
                'comment_count': 'Comments'
            }
        )
        
        # Add trendline
        if len(df) > 1:
            z = np.polyfit(df['view_count'], df['engagement_rate'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=df['view_count'].sort_values(),
                    y=p(df['view_count'].sort_values()),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
        
        fig.update_layout(
            xaxis_title="Views",
            yaxis_title="Engagement Rate"
        )
        
        return fig
    
    def create_upload_day_analysis(self, df):
        """
        Create bar chart of performance by upload day
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_performance = df.groupby('publish_day').agg({
            'view_count': 'mean',
            'engagement_rate': 'mean'
        }).reindex(day_order)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Views bar chart
        fig.add_trace(
            go.Bar(
                x=day_performance.index,
                y=day_performance['view_count'],
                name='Average Views',
                marker_color=self.primary_color
            ),
            secondary_y=False,
        )
        
        # Engagement line chart
        fig.add_trace(
            go.Scatter(
                x=day_performance.index,
                y=day_performance['engagement_rate'],
                mode='lines+markers',
                name='Average Engagement Rate',
                line=dict(color='red', width=3)
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Day of Week")
        fig.update_yaxes(title_text="Average Views", secondary_y=False)
        fig.update_yaxes(title_text="Average Engagement Rate", secondary_y=True)
        
        fig.update_layout(
            title_text="Performance by Upload Day",
            hovermode='x unified'
        )
        
        return fig
    
    def create_upload_hour_analysis(self, df):
        """
        Create bar chart of performance by upload hour
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        hour_performance = df.groupby('publish_hour').agg({
            'view_count': 'mean',
            'engagement_rate': 'mean'
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Views bar chart
        fig.add_trace(
            go.Bar(
                x=hour_performance.index,
                y=hour_performance['view_count'],
                name='Average Views',
                marker_color=self.primary_color
            ),
            secondary_y=False,
        )
        
        # Engagement line chart
        fig.add_trace(
            go.Scatter(
                x=hour_performance.index,
                y=hour_performance['engagement_rate'],
                mode='lines+markers',
                name='Average Engagement Rate',
                line=dict(color='red', width=3)
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Hour of Day (24h format)")
        fig.update_yaxes(title_text="Average Views", secondary_y=False)
        fig.update_yaxes(title_text="Average Engagement Rate", secondary_y=True)
        
        fig.update_layout(
            title_text="Performance by Upload Hour",
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_heatmap(self, df):
        """
        Create heatmap of performance by day and hour
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        # Create pivot table
        pivot_data = df.pivot_table(
            values='view_count',
            index='publish_day',
            columns='publish_hour',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(day_order)
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Average Views"),
            title="Performance Heatmap: Views by Day and Hour"
        )
        
        return fig
    
    def create_duration_analysis(self, df):
        """
        Create analysis of video duration vs performance
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Box plot
        """
        if 'duration_seconds' not in df.columns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Duration data not available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Create duration bins
        duration_bins = pd.cut(
            df['duration_seconds'],
            bins=[0, 60, 300, 600, 1200, float('inf')],
            labels=['0-1min', '1-5min', '5-10min', '10-20min', '20min+']
        )
        
        df_with_bins = df.copy()
        df_with_bins['duration_category'] = duration_bins
        
        fig = px.box(
            df_with_bins,
            x='duration_category',
            y='view_count',
            title="Video Performance by Duration",
            labels={'duration_category': 'Video Duration', 'view_count': 'Views'}
        )
        
        return fig
    
    def create_consistency_chart(self, df):
        """
        Create chart showing upload consistency over time
        
        Args:
            df (pd.DataFrame): Video data DataFrame
            
        Returns:
            plotly.graph_objects.Figure: Line chart
        """
        df_sorted = df.sort_values('published_at').copy()
        
        # Calculate time differences between uploads
        df_sorted['days_since_last'] = df_sorted['published_at'].diff().dt.days
        
        fig = px.line(
            df_sorted,
            x='published_at',
            y='days_since_last',
            title="Upload Consistency Over Time",
            labels={
                'published_at': 'Upload Date',
                'days_since_last': 'Days Since Last Upload'
            }
        )
        
        # Add average line
        if not df_sorted['days_since_last'].dropna().empty:
            avg_days = df_sorted['days_since_last'].mean()
            fig.add_hline(
                y=avg_days,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Average: {avg_days:.1f} days"
            )
        
        return fig
