import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import imageio
from PIL import Image
import io
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

class ModelVisualizer:
    def __init__(self, predictions, true_values, spatial_coords, redshifts=None, cosmological_time=None, subhalo_coords=None, save_dir=None):
        """
        Initialize the ModelVisualizer.

        Args:
            predictions (numpy.ndarray): Model predictions.
            true_values (numpy.ndarray): True target values.
            spatial_coords (numpy.ndarray): Spatial coordinates of the data points.
            redshifts (numpy.ndarray, optional): Redshift values for each data point. Defaults to None.
            cosmological_time (numpy.ndarray, optional): Cosmological time values. Defaults to None.
            subhalo_coords (numpy.ndarray, optional): Spatial coordinates of subhalos. Defaults to None.
            save_dir (str, optional): Directory to save visualizations. Defaults to None.
        """
        
        self.predictions = np.asarray(predictions).reshape(-1)  # Reshape predictions
        self.true_values = np.asarray(true_values).reshape(-1)  # Reshape true_values
        self.spatial_coords = np.asarray(spatial_coords)
        
        if len(self.predictions) != len(self.true_values) or len(self.predictions) != len(self.spatial_coords):
            logger.error("Input arrays must have the same length")
            raise ValueError("Input arrays must have the same length")
            
        if self.spatial_coords.shape[1] != 3:
            logger.error("Spatial coordinates must be 3-dimensional")
            raise ValueError("Spatial coordinates must be 3-dimensional")
            
        self.redshifts = np.asarray(redshifts) if redshifts is not None else None
        self.cosmological_time = np.asarray(cosmological_time) if cosmological_time is not None else None
        self.subhalo_coords = np.asarray(subhalo_coords) if subhalo_coords is not None else None
        self.save_dir = save_dir
        
        # Calculate min/max radius and spatial limits
        radii = np.linalg.norm(self.spatial_coords, axis=1)
        self.min_radius = np.min(radii[radii > 0])  # Avoid zero radius
        self.max_radius = np.max(radii)
        
        # Calculate spatial limits for consistent visualization
        self.x_limits = [np.min(self.spatial_coords[:, 0]), np.max(self.spatial_coords[:, 0])]
        self.y_limits = [np.min(self.spatial_coords[:, 1]), np.max(self.spatial_coords[:, 1])]
        self.z_limits = [np.min(self.spatial_coords[:, 2]), np.max(self.spatial_coords[:, 2])]
        
        # Expand limits by 10% for better visualization
        self.x_range = self.x_limits[1] - self.x_limits[0]
        self.y_range = self.y_limits[1] - self.y_limits[0]
        self.z_range = self.z_limits[1] - self.z_limits[0]
        
        self.x_limits = [self.x_limits[0] - 0.1 * self.x_range, self.x_limits[1] + 0.1 * self.x_range]
        self.y_limits = [self.y_limits[0] - 0.1 * self.y_range, self.y_limits[1] + 0.1 * self.y_range]
        self.z_limits = [self.z_limits[0] - 0.1 * self.z_range, self.z_limits[1] + 0.1 * self.z_range]

    def __calculate_density_profile(self, weight_function=None):
        r_bins = np.logspace(np.log10(self.min_radius), np.log10(self.max_radius), 100)
        density_profile = np.zeros_like(r_bins)
        radii = np.linalg.norm(self.spatial_coords, axis=1)
        
        for i, r in enumerate(r_bins):
            mask = radii <= r
            if np.sum(mask) == 0:
                continue  # Skip empty bins
                
            if weight_function and self.redshifts is not None:
                weights = weight_function(self.redshifts[mask])
                density_profile[i] = np.sum(self.predictions[mask] * weights) / (4/3 * np.pi * r**3)
            else:
                density_profile[i] = np.sum(self.predictions[mask]) / (4/3 * np.pi * r**3)
                
        return r_bins, density_profile

    def __plot_density_profile(self,r_bins : np.ndarray, profile: np.ndarray, title:str, yaxis_title: str) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=r_bins,
                y=profile,
                mode='lines',
                name=r"$\text{" + title + r"}$",  # LaTeX title in legend
                line=dict(width=2)
            )
        )

        fig.update_layout(
            title=r"$\text{" + title + r"}$",  # LaTeX formatted title
            xaxis_title=r"$\text{Radius (Mpc)}$",  # LaTeX x-axis title
            yaxis_title=r"$\text{" + yaxis_title + r"}$",  # LaTeX formatted y-axis title
            xaxis=dict(type="log", title_font=dict(size=16)),
            yaxis=dict(type="log", title_font=dict(size=16)),
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=14)
        )

        return fig
    
    def density_profile(self):
        try:
            r_bins, density_profile = self.__calculate_density_profile()
            return self.__plot_density_profile(r_bins, density_profile, 'Dark Matter Halos Density Profile', 'Density (Msun/Mpc^3)')
        except Exception as e:
            logger.error(f"Error in density profile calculation: {e}")
            return None
    
    def cuspy_density_profile(self):
        try:
            r_bins, cuspy_density_profile = self.__calculate_density_profile(weight_function=lambda z: z)
            return self.__plot_density_profile(r_bins, cuspy_density_profile, 'Dark Matter Halos Cuspy Density Profile', 'Cuspy Density (Msun/Mpc^3)')
        except Exception as e:
            logger.error(f"Error in cuspy density profile calculation: {e}")
            return None
    
    def nfw_density_profile(self):
        try:
            r_bins, nfw_density_profile = self.__calculate_density_profile(weight_function=lambda z: z**2)
            return self.__plot_density_profile(r_bins, nfw_density_profile, 'Dark Matter Halos nfw Density Profile', 'nfw Density (Msun/Mpc^3)')
        except Exception as e:
            logger.error(f"Error in nfw density profile calculation: {e}")
            return None

    def __plot_all_density_profiles(self, r_bins, profiles):
        fig = go.Figure()
        
        for name, profile in profiles.items():
            fig.add_trace(
                go.Scatter(
                    x=r_bins,
                    y=profile,
                    mode='lines',
                    name=f"$\\text{{{name}}}$",
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=r"$\text{Dark Matter Halos Density Profiles}$",
            xaxis_title=r"$\text{Radius (Mpc)}$",
            yaxis_title=r"$\text{Density (M_{\odot} / Mpc^3)}$",
            xaxis=dict(type="log", title_font=dict(size=16)),  # Log scale for x-axis
            yaxis=dict(type="log", title_font=dict(size=16)),  # Log scale for y-axis
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=14)
        )
        
        return fig

    def all_density_profiles(self):
        try:
            r_bins, density_profile = self.__calculate_density_profile()
            _, cuspy_density_profile = self.__calculate_density_profile(weight_function=lambda z: z)
            _, nfw_density_profile = self.__calculate_density_profile(weight_function=lambda z: z**2)

            profiles = {
                'Standard Density Profile': density_profile,
                'Cuspy Density Profile': cuspy_density_profile,
                'NFW Density Profile': nfw_density_profile
            }

            return self.__plot_all_density_profiles(r_bins, profiles)
        except Exception as e:
            logger.error(f"Error in plotting all density profiles: {e}")
            return None

    def visualize_dm_distribution(self, a=1.0, b=0.8, c=0.6):
        """
        Visualize the distribution of DM values using a triaxial ellipsoid and redshift evolution,
        separated into three parts: Main DM Distribution, Triaxial Model, and Evolution View.
        
        Args:
            a (float, optional): Major axis length. Defaults to 1.0.
            b (float, optional): Intermediate axis length. Defaults to 0.8.
            c (float, optional): Minor axis length. Defaults to 0.6.
        
        Returns:
            plotly.graph_objects.Figure: The visualization figure or None if error occurs
        """
        try:
            # Input validation
            if self.spatial_coords.shape[1] != 3:
                logger.error("Spatial coordinates should have 3 dimensions.")
                raise ValueError("Spatial coordinates should have 3 dimensions.")
            
            # Create subplots with proper spacing
            fig = sp.make_subplots(
                rows=1,
                cols=3,
                specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
                subplot_titles=[
                    r"$\text{Main DM Distribution}$", 
                    r"$\text{Triaxial Model}$", 
                    r"$\text{Evolution View}$"
                ],
                horizontal_spacing=0.02,
                vertical_spacing=0.1
            )
            
            # Calculate color values for visualization
            color_vals = np.linalg.norm(self.spatial_coords, axis=1)
            norm_predictions = (self.predictions - np.min(self.predictions)) / \
                            (np.max(self.predictions) - np.min(self.predictions))
            
            # Create initial traces for each subplot
            # 1. Main DM Distribution
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[:, 0],
                    y=self.spatial_coords[:, 1],
                    z=self.spatial_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=color_vals,
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title=r"$\text{Distance from Center}$", x=0.28)
                    ),
                    name=r"$\text{DM Particles}$"
                ),
                row=1, col=1
            )
            
            # 2. Triaxial Model
            center = np.mean(self.spatial_coords, axis=0)
            surfaces = self.__create_triaxial_surfaces(a, b, c, center)

            for surface in surfaces:
                fig.add_trace(surface, row=1, col=2)
            
            # 3. Evolution View (initial state)
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[:, 0],
                    y=self.spatial_coords[:, 1],
                    z=self.spatial_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=norm_predictions,
                        colorscale='Inferno',
                        opacity=0.8,
                        colorbar=dict(title=r"$\text{Normalized Predictions}$", x=1.0)
                    ),
                    name=r"$\text{Predictions}$"
                ),
                row=1, col=3
            )
            
            # Animation setup if redshifts available
            frames = []
            if self.redshifts is not None:
                unique_redshifts = np.sort(np.unique(self.redshifts))[::-1]
                
                for z in unique_redshifts:
                    mask = (self.redshifts == z)
                    
                    # Create frame data for all three subplots
                    frame_data = [
                        # Subplot 1: DM Distribution
                        go.Scatter3d(
                            x=self.spatial_coords[mask, 0],
                            y=self.spatial_coords[mask, 1],
                            z=self.spatial_coords[mask, 2],
                            mode='markers',
                            marker=dict(
                                size=3,
                                color=color_vals[mask],
                                colorscale='Viridis',
                                opacity=0.8
                            )
                        ),
                        
                        # Subplot 2: Triaxial Model (keeps surfaces)
                        *self.__create_triaxial_surfaces(a, b, c, center),
                        
                        # Subplot 3: Evolution View
                        go.Scatter3d(
                            x=self.spatial_coords[mask, 0],
                            y=self.spatial_coords[mask, 1],
                            z=self.spatial_coords[mask, 2],
                            mode='markers',
                            marker=dict(
                                size=3,
                                color=norm_predictions[mask],
                                colorscale='Inferno',
                                opacity=0.8
                            )
                        )
                    ]
                    
                    frames.append(go.Frame(
                        data=frame_data,
                        name=f"z={z:.2f}",
                        traces=[0, 2, 3, 4, 5]  # Update all traces except static surfaces
                    ))
            
            # Layout configuration with axis limits
            fig.update_layout(
                title_text=r"$\text{Dark Matter Halo Distribution with Triaxial Model}$",
                width=1800,
                height=700,
                margin=dict(l=50, r=50, b=50, t=100),
                scene1=dict(
                    xaxis_title=r"$X\ (\text{Mpc})$",
                    yaxis_title=r"$Y\ (\text{Mpc})$",
                    zaxis_title=r"$Z\ (\text{Mpc})$",
                    aspectmode='cube',
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                ),
                scene2=dict(
                    xaxis_title=r"$X\ (\text{Mpc})$",
                    yaxis_title=r"$Y\ (\text{Mpc})$",
                    zaxis_title=r"$Z\ (\text{Mpc})$",
                    aspectmode='cube',
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                ),
                scene3=dict(
                    xaxis_title=r"$X\ (\text{Mpc})$",
                    yaxis_title=r"$Y\ (\text{Mpc})$",
                    zaxis_title=r"$Z\ (\text{Mpc})$",
                    aspectmode='cube',
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                ),
                showlegend=True,
                legend=dict(x=1.05, y=0.5),
                template='plotly_white'
            )
            
            # Add animation controls if we have frames
            if frames:
                fig.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'buttons': [
                            {
                                'args': [None, {
                                    'frame': {'duration': 300, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 100}
                                }],
                                'label': 'Play',
                                'method': 'animate'
                            },
                            {
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }],
                                'label': 'Pause',
                                'method': 'animate'
                            }
                        ],
                        'x': 0.1,
                        'y': 0,
                        'xanchor': 'right',
                        'yanchor': 'top'
                    }],
                    sliders=[{
                        'active': 0,
                        'currentvalue': {'prefix': 'Redshift: '},
                        'steps': [
                            {
                                'args': [[frame.name], {
                                    'frame': {'duration': 300, 'redraw': True},
                                    'mode': 'immediate'
                                }],
                                'label': frame.name,
                                'method': 'animate'
                            }
                            for frame in frames
                        ]
                    }]
                )
                fig.frames = frames
            
            # Save or display the figure
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                plot_path = os.path.join(self.save_dir, 'dm_distribution.html')
                fig.write_html(
                    plot_path,
                    include_mathjax='cdn'
                )
                logger.info(f"Visualization saved to {plot_path}")
                
                # Create GIF from frames if available
                if frames:
                    self.create_visualization_gif(fig, frames, os.path.join(self.save_dir, 'dm_distribution.gif'))
            else:
                fig.show()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error in visualization: {e}", exc_info=True)
            return None
    
    def __create_triaxial_surfaces(self, a, b, c, center, n_points=50, opacity=0.3):
        """
        Create triaxial model surfaces
        
        Args:
            a (float): Major axis length
            b (float): Minor axis length
            c (float): Short axis length
            center (float): Center of the model
            n_points (int, optional): Number of points to use for surface approximation. Defaults to 50.
            opacity (float, optional): Opacity of surfaces. Defaults to 0.3.
        """
        phi = np.linspace(0, 2*np.pi, n_points)
        theta = np.linspace(-np.pi/2, np.pi/2, n_points)
        phi, theta = np.meshgrid(phi, theta)
        
        surfaces = []
        
        # Outer shell
        x_outer = a * np.cos(theta) * np.cos(phi) + center[0]
        y_outer = b * np.cos(theta) * np.sin(phi) + center[1]
        z_outer = c * np.sin(theta) + center[2]
        
        surfaces.append(
            go.Surface(
                x=x_outer,
                y=y_outer,
                z=z_outer,
                opacity=opacity,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title=r"$\text{Outer Shell Density}$"),
                name=r"$\text{Outer Shell}$"
            )
        )
        
        # Inner shell
        x_inner = 0.7 * a * np.cos(theta) * np.cos(phi) + center[0]
        y_inner = 0.7 * b * np.cos(theta) * np.sin(phi) + center[1]
        z_inner = 0.7 * c * np.sin(theta) + center[2]
        
        surfaces.append(
            go.Surface(
                x=x_inner,
                y=y_inner,
                z=z_inner,
                opacity=opacity,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title=r"$\text{Inner Shell Density}$"),
                name=r"$\text{Inner Shell}$"
            )
        )
        
        return surfaces
    
    def create_visualization_gif(self, fig, frames, save_path):
        """
        Create an animated GIF from a figure with frames.
        
        Args:
            fig (plotly.graph_objects.Figure): Figure with frames
            frames (list): List of frames
            save_path (str): Path to save the GIF
        """
        try:
            # Convert frames to images
            image_frames = []
            for frame in frames:
                # Create a copy of the figure
                fig_copy = go.Figure(fig)
                
                # Update with frame data
                for i, trace in enumerate(frame.data):
                    if i < len(fig_copy.data):
                        fig_copy.data[i].update(trace)
                
                # Update title if present
                if hasattr(frame, 'name') and frame.name:
                    current_title = fig_copy.layout.title.text
                    if current_title:
                        base_title = current_title.split(" at")[0]
                        fig_copy.update_layout(title=f"{base_title} at {frame.name}")
                
                # Convert to image
                img_bytes = fig_copy.to_image(format="png", width=1800, height=700)
                img = Image.open(io.BytesIO(img_bytes))
                image_frames.append(np.array(img))
            
            # Save as GIF
            imageio.mimsave(save_path, image_frames, duration=300)
            logger.info(f"Visualization GIF saved to {save_path}")
        except Exception as e:
            logger.error(f"Error creating GIF: {e}", exc_info=True)
    
    def create_evolution_gif(self, save_path):
        """Create a GIF of the evolution of the model
        
        Args:
        save_path (str): Path to save the GIF
        """
        unique_redshifts = np.sort(np.unique(self.redshifts))[::-1]
        frames = []
        
        for z in unique_redshifts:
            mask = (self.redshifts == z)
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=self.spatial_coords[mask, 0],
                    y=self.spatial_coords[mask, 1],
                    z=self.spatial_coords[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=self.predictions[mask],
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title=r"$\text{Density}$")
                    )
                )
            ])
        
            fig.update_layout(
                title=r"$\text{Dark Matter Distribution at } z=" + f"{z:.2f}$",
                width=800,
                height=800,
                template='plotly_white',
                scene=dict(
                    xaxis_title=r"$X \, (\text{Mpc})$",
                    yaxis_title=r"$Y \, (\text{Mpc})$",
                    zaxis_title=r"$Z \, (\text{Mpc})$",
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                )
            )
            
            # Convert to image
            img_bytes = fig.to_image(format="png")
            img = Image.open(io.BytesIO(img_bytes))
            frames.append(np.array(img))
        
        # Save as GIF
        imageio.mimsave(save_path, frames, duration=100)
        logger.info(f"Evolution GIF saved to {save_path}")

    def visualize_triaxial_with_subhalos(self, opacity=0.2):
        """
        Visualize triaxial model with subhalo distribution inside and dark matter particles
        
        Args:
            opacity (float, optional): Opacity of surfaces. Defaults to 0.2.
        
        Returns:
            plotly.graph_objects.Figure: The visualization figure or None if error occurs
        """
        try:
            # Check if subhalo coordinates are available
            if self.subhalo_coords is None:
                logger.warning("No subhalo coordinates provided. Using random subset of particles instead.")
                # If no subhalo coordinates, use a random subset of particles as subhalos
                rng = np.random.RandomState(42)  # Fixed seed for reproducibility
                n_subhalos = min(int(len(self.spatial_coords) * 0.05), 1000)  # 5% of particles or max 1000
                subhalo_indices = rng.choice(len(self.spatial_coords), n_subhalos, replace=False)
                self.subhalo_coords = self.spatial_coords[subhalo_indices]
            
            # Fit triaxial model to main distribution
            a, b, c, center = self.__fit_triaxial_model()
            if a is None:
                logger.error("Failed to fit triaxial model")
                return None
            
            # Create figure
            fig = go.Figure()
            
            # Add inner and outer shells with increased transparency
            surfaces = self.__create_triaxial_surfaces(a, b, c, center, opacity=opacity)
            for surface in surfaces:
                fig.add_trace(surface)
            
            # Add dark matter particles
            particle_distances = np.linalg.norm(self.spatial_coords - center, axis=1)
            norm_distances = (particle_distances - np.min(particle_distances)) / (np.max(particle_distances) - np.min(particle_distances))
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[:, 0],
                    y=self.spatial_coords[:, 1],
                    z=self.spatial_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=norm_distances,
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(
                            title=r"$\text{Distance from Center}$",
                            x=0.85
                        )
                    ),
                    name=r"$\text{DM Particles}$"
                )
            )
            
            # Add subhalos
            subhalo_distances = np.linalg.norm(self.subhalo_coords - center, axis=1)
            subhalo_sizes = 5 + 10 * (1 - subhalo_distances / np.max(subhalo_distances))  # Larger subhalos near center
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.subhalo_coords[:, 0],
                    y=self.subhalo_coords[:, 1],
                    z=self.subhalo_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=subhalo_sizes,
                        color=subhalo_distances,
                        colorscale='Plasma',
                        opacity=0.8,
                        symbol='diamond',
                        colorbar=dict(
                            title=r"$\text{Subhalo Distance}$",
                            x=1.0
                        )
                    ),
                    name=r"$\text{Subhalos}$"
                )
            )
            
            # Animation setup if redshifts available
            frames = []
            if self.redshifts is not None:
                unique_redshifts = np.sort(np.unique(self.redshifts))[::-1]
                
                for z in unique_redshifts:
                    # Filter data by redshift
                    dm_mask = (self.redshifts == z)
                    
                    # For subhalos, either filter by redshift if available or show all
                    if hasattr(self, 'subhalo_redshifts') and self.subhalo_redshifts is not None:
                        subhalo_mask = (self.subhalo_redshifts == z)
                        current_subhalos = self.subhalo_coords[subhalo_mask]
                        current_subhalo_distances = subhalo_distances[subhalo_mask]
                        current_subhalo_sizes = subhalo_sizes[subhalo_mask]
                    else:
                        current_subhalos = self.subhalo_coords
                        current_subhalo_distances = subhalo_distances
                        current_subhalo_sizes = subhalo_sizes
                    
                    # Create frame with updated data
                    frame_data = [
                        # Keep original surfaces
                        *surfaces,
                        
                        # Update DM particles
                        go.Scatter3d(
                            x=self.spatial_coords[dm_mask, 0],
                            y=self.spatial_coords[dm_mask, 1],
                            z=self.spatial_coords[dm_mask, 2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=norm_distances[dm_mask],
                                colorscale='Viridis',
                                opacity=0.6
                            )
                        ),
                        
                        # Update subhalos
                        go.Scatter3d(
                            x=current_subhalos[:, 0],
                            y=current_subhalos[:, 1],
                            z=current_subhalos[:, 2],
                            mode='markers',
                            marker=dict(
                                size=current_subhalo_

def visualize_triaxial_with_subhalos(self, opacity=0.2):
        """
        Visualize triaxial model with subhalo distribution inside and dark matter particles
        
        Args:
            opacity (float, optional): Opacity of surfaces. Defaults to 0.2.
        
        Returns:
            plotly.graph_objects.Figure: The visualization figure or None if error occurs
        """
        try:
            # Check if subhalo coordinates are available
            if self.subhalo_coords is None:
                logger.warning("No subhalo coordinates provided. Using random subset of particles instead.")
                # If no subhalo coordinates, use a random subset of particles as subhalos
                rng = np.random.RandomState(42)  # Fixed seed for reproducibility
                n_subhalos = min(int(len(self.spatial_coords) * 0.05), 1000)  # 5% of particles or max 1000
                subhalo_indices = rng.choice(len(self.spatial_coords), n_subhalos, replace=False)
                self.subhalo_coords = self.spatial_coords[subhalo_indices]
            
            # Fit triaxial model to main distribution
            a, b, c, center = self.__fit_triaxial_model()
            if a is None:
                logger.error("Failed to fit triaxial model")
                return None
            
            # Create figure
            fig = go.Figure()
            
            # Add inner and outer shells with increased transparency
            surfaces = self.__create_triaxial_surfaces(a, b, c, center, opacity=opacity)
            for surface in surfaces:
                fig.add_trace(surface)
            
            # Add dark matter particles
            particle_distances = np.linalg.norm(self.spatial_coords - center, axis=1)
            norm_distances = (particle_distances - np.min(particle_distances)) / (np.max(particle_distances) - np.min(particle_distances))
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[:, 0],
                    y=self.spatial_coords[:, 1],
                    z=self.spatial_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=norm_distances,
                        colorscale='Viridis',
                        opacity=0.6,
                        colorbar=dict(
                            title=r"$\text{Distance from Center}$",
                            x=0.85
                        )
                    ),
                    name=r"$\text{DM Particles}$"
                )
            )
            
            # Add subhalos
            subhalo_distances = np.linalg.norm(self.subhalo_coords - center, axis=1)
            subhalo_sizes = 5 + 10 * (1 - subhalo_distances / np.max(subhalo_distances))  # Larger subhalos near center
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.subhalo_coords[:, 0],
                    y=self.subhalo_coords[:, 1],
                    z=self.subhalo_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=subhalo_sizes,
                        color=subhalo_distances,
                        colorscale='Plasma',
                        opacity=0.8,
                        symbol='diamond',
                        colorbar=dict(
                            title=r"$\text{Subhalo Distance}$",
                            x=1.0
                        )
                    ),
                    name=r"$\text{Subhalos}$"
                )
            )
            
            # Animation setup if redshifts available
            frames = []
            if self.redshifts is not None:
                unique_redshifts = np.sort(np.unique(self.redshifts))[::-1]
                
                for z in unique_redshifts:
                    # Filter data by redshift
                    dm_mask = (self.redshifts == z)
                    
                    # For subhalos, either filter by redshift if available or show all
                    if hasattr(self, 'subhalo_redshifts') and self.subhalo_redshifts is not None:
                        subhalo_mask = (self.subhalo_redshifts == z)
                        current_subhalos = self.subhalo_coords[subhalo_mask]
                        current_subhalo_distances = subhalo_distances[subhalo_mask]
                        current_subhalo_sizes = subhalo_sizes[subhalo_mask]
                    else:
                        current_subhalos = self.subhalo_coords
                        current_subhalo_distances = subhalo_distances
                        current_subhalo_sizes = subhalo_sizes
                    
                    # Create frame with updated data
                    frame_data = [
                        # Keep original surfaces
                        *surfaces,
                        
                        # Update DM particles
                        go.Scatter3d(
                            x=self.spatial_coords[dm_mask, 0],
                            y=self.spatial_coords[dm_mask, 1],
                            z=self.spatial_coords[dm_mask, 2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=norm_distances[dm_mask],
                                colorscale='Viridis',
                                opacity=0.6
                            )
                        ),
                        
                        # Update subhalos
                        go.Scatter3d(
                            x=current_subhalos[:, 0],
                            y=current_subhalos[:, 1],
                            z=current_subhalos[:, 2],
                            mode='markers',
                            marker=dict(
                                size=current_subhalo_sizes,
                                color=current_subhalo_distances,
                                colorscale='Plasma',
                                opacity=0.8,
                                symbol='diamond'
                                
def __fit_triaxial_model(self):
        """
        Fit predicted positions into a triaxial ellipsoid model.
        Returns fitted axis lengths a, b, c and center.
        """
        try:
            # Use predicted positions
            coords = self.spatial_coords
            if coords.shape[1] != 3:
                raise ValueError("Spatial coordinates must be 3D.")

            # Compute covariance matrix
            centered = coords - np.mean(coords, axis=0)
            cov = np.cov(centered, rowvar=False)
            
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Axis lengths proportional to sqrt of eigenvalues
            axis_lengths = 2 * np.sqrt(eigvals)
            a, b, c = sorted(axis_lengths, reverse=True)

            logger.info(f"Fitted axes lengths: a={a:.3f}, b={b:.3f}, c={c:.3f}")
            return a, b, c, np.mean(coords, axis=0)
        
        except Exception as e:
            logger.error(f"Error in fitting triaxial model: {e}")
            return None, None, None, None
        
    def visualize_fitted_triaxial_model(self):
        """
        Visualize the predicted DM halo positions fitted into a triaxial ellipsoid model.
        """
        try:
            a, b, c, center = self.__fit_triaxial_model()
            if a is None:
                return None
            
            surfaces = self.__create_triaxial_surfaces(a, b, c, center)

            distances = np.linalg.norm(self.spatial_coords - center, axis=1)
            normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

            scatter = go.Scatter3d(
                x=self.spatial_coords[:, 0],
                y=self.spatial_coords[:, 1],
                z=self.spatial_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=normalized_distances,  # Color based on distance from center
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title=r"$\text{Predicted Mass}$")
                ),
                name=r'$\text{Particle Distribution}$'
            )
            
            fig_data = surfaces + [scatter]
            
            fig = go.Figure(data=fig_data)
            fig.update_layout(
                title=r"$\text{Fitted Triaxial Model with Particle Distribution}$",
                scene=dict(
                    xaxis_title=r"$X \, (\text{Mpc})$",
                    yaxis_title=r"$Y \, (\text{Mpc})$",
                    zaxis_title=r"$Z \, (\text{Mpc})$",
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                ),
                width=800,
                height=800,
                template='plotly_white'
            )
            
            if self.save_dir:
                path = os.path.join(self.save_dir, 'fitted_triaxial_model.html')
                fig.write_html(
                    path,
                    include_mathjax='cdn'
                )
                logger.info(f"Fitted visualization saved to {path}")

            return fig, a, b, c

        except Exception as e:
            logger.error(f"Error in triaxial visualization: {e}")
            return None
    
    def visualize_cosmological_evolution(self):
        """
        Visualize dark matter evolution based on cosmological time-series.
        
        Returns:
            plotly.graph_objects.Figure: The visualization figure or None if error occurs
        """
        try:
            if self.cosmological_time is None:
                logger.warning("No cosmological time data provided. Using redshifts instead if available.")
                if self.redshifts is None:
                    logger.error("Neither cosmological time nor redshift data available.")
                    return None
                
                # Create a proxy time variable from redshift for visualization 
                # (higher redshift = earlier time)
                time_variable = -self.redshifts  # Negative so higher redshift = earlier time
                time_label = "Redshift z"
                frame_prefix = "z="
            else:
                time_variable = self.cosmological_time
                time_label = "Cosmological Time (Gyr)"
                frame_prefix = "t="
            
            # Get unique time points for frames, sorted chronologically
            unique_times = np.sort(np.unique(time_variable))
            if time_variable is -self.redshifts:  # If using redshift, reverse order
                unique_times = unique_times[::-1]
            
            # Create figure
            fig = go.Figure()
            
            # Add initial frame (earliest time)
            initial_time = unique_times[0]
            mask = (time_variable == initial_time)
            
            # Color by prediction values
            norm_predictions = (self.predictions - np.min(self.predictions)) / \
                            (np.max(self.predictions) - np.min(self.predictions))
            
            # Add initial trace
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[mask, 0],
                    y=self.spatial_coords[mask, 1],
                    z=self.spatial_coords[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=norm_predictions[mask],
                        colorscale='Plasma',
                        opacity=0.8,
                        colorbar=dict(title=r"$\text{Normalized Density}$")
                    ),
                    name=r"$\text{DM Distribution}$"
                )
            )
            
            # Create frames for animation
            frames = []
            
            for t in unique_times:
                mask = (time_variable == t)
                
                # Skip if no data for this time
                if np.sum(mask) == 0:
                    continue
                    
                # Add density contours or surfaces to show clustering evolution
                # This is a placeholder for what would be more complex analysis
                
                frame_data = [
                    go.Scatter3d(
                        x=self.spatial_coords[mask, 0],
                        y=self.spatial_coords[mask, 1],
                        z=self.spatial_coords[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=norm_predictions[mask],
                            colorscale='Plasma', 
                            opacity=0.8
                        )
                    )
                ]
                
                # Format time value for display
                if frame_prefix == "z=":
                    time_display = f"{-t:.2f}"  # Convert back to positive redshift
                else:
                    time_display = f"{t:.2f}"
                
                frames.append(go.Frame(
                    data=frame_data,
                    name=f"{frame_prefix}{time_display}",
                    traces=[0]
                ))
            
            # Layout with animation controls
            fig.update_layout(
                title=r"$\text{Dark Matter Evolution Through Cosmic Time}$",
                scene=dict(
                    xaxis_title=r"$X \, (\text{Mpc})$",
                    yaxis_title=r"$Y \, (\text{Mpc})$",
                    zaxis_title=r"$Z \, (\text{Mpc})$",
                    aspectmode='cube',
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                ),
                updatemenus=[{
                    'type': 'buttons',
                    'buttons': [
                        {
                            'args': [None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'x': 0.1,
                    'y': 0,
                    'xanchor': 'right',
                    'yanchor': 'top'
                }],
                sliders=[{
                    'active': 0,
                    'currentvalue': {'prefix': f'{time_label}: '},
                    'steps': [
                        {
                            'args': [[frame.name], {'frame': {'duration': 300}, 'mode': 'immediate'}],
                            'label': frame.name,
                            'method': 'animate'
                        } for frame in frames
                    ]
                }],
                width=1000,
                height=800,
                template='plotly_white'
            )
            
            fig.frames = frames
            
            # Save figure and create GIF
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                html_path = os.path.join(self.save_dir, 'cosmological_evolution.html')
                fig.write_html(html_path, include_mathjax='cdn')
                
                gif_path = os.path.join(self.save_dir, 'cosmological_evolution.gif')
                self.create_visualization_gif(fig, frames, gif_path)
                
                logger.info(f"Cosmological evolution visualization saved to {html_path} and {gif_path}")
            
            return fig
        
        except Exception as e:
            logger.error(f"Error in cosmological evolution visualization: {e}", exc_info=True)
            return None
    
    def visualize_hierarchical_structure(self):
        """
        Create a hierarchical visualization showing main halo, subhalos, and DM particles
        with clustering metrics over cosmic time.
        
        Returns:
            plotly.graph_objects.Figure: The visualization figure or None if error occurs
        """
        try:
            # Fit triaxial model
            a, b, c, center = self.__fit_triaxial_model()
            if a is None:
                return None
            
            # Create figure with 4 subplots:
            # 1. 3D view of hierarchical structure
            # 2. Correlation of subhalo distance vs. DM density
            # 3. Subhalo occupation vs radius
            # 4. Clustering metrics over time
            
            fig = sp.make_subplots(
                rows=2, 
                cols=2,
                specs=[
                    [{"type": "scatter3d"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ],
                subplot_titles=[
                    "Hierarchical Structure", 
                    "Subhalo Distance vs. DM Density",
                    "Subhalo Occupation",
                    "Clustering Metrics"
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 1. 3D Hierarchical Structure
            # Use subhalo coordinates if available, otherwise generate some
            if self.subhalo_coords is None:
                # Generate random subhalos within the triaxial structure
                rng = np.random.RandomState(42)
                n_subhalos = min(int(len(self.spatial_coords) * 0.02), 50000)
                subhalo_indices = rng.choice(len(self.spatial_coords), n_subhalos, replace=False)
                self.subhalo_coords = self.spatial_coords[subhalo_indices]
            
            # Calculate distances from center
            dm_distances = np.linalg.norm(self.spatial_coords - center, axis=1)
            subhalo_distances = np.linalg.norm(self.subhalo_coords - center, axis=1)
            
            # Scale marker sizes based on distance from center (closer = larger)
            subhalo_sizes = 5 + 15 * (1 - subhalo_distances / np.max(subhalo_distances))
            
            # Add main halo (triaxial model)
            surfaces = self.__create_triaxial_surfaces(a, b, c, center, opacity=0.15)
            for surface in surfaces:
                fig.add_trace(surface, row=1, col=1)
            
            # Add dark matter particles with subsampling for better visualization
            sample_size = min(len(self.spatial_coords), 50000)  # Limit number for performance
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(len(self.spatial_coords), sample_size, replace=False)
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[sample_indices, 0],
                    y=self.spatial_coords[sample_indices, 1],
                    z=self.spatial_coords[sample_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=self.predictions[sample_indices],
                        colorscale='Viridis',
                        opacity=0.5
                    ),
                    name="DM Particles"
                ),
                row=1, col=1
            )
            
            # Add subhalos
            fig.add_trace(
                go.Scatter3d(
                    x=self.subhalo_coords[:, 0],
                    y=self.subhalo_coords[:, 1],
                    z=self.subhalo_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=subhalo_sizes,
                        color=subhalo_distances,
                        colorscale='Plasma',
                        opacity=0.8,
                        symbol='diamond'
                    ),
                    name="Subhalos"
                ),
                row=1, col=1
            )
            
            # 2. Subhalo Distance vs. DM Density
            # For each subhalo, compute local DM density
            subhalo_local_density = []
            radius_search = (a + b + c) / 30  # Use a fraction of the triaxial size
            
            for subhalo in self.subhalo_coords:
                # Find DM particles near this subhalo
                distances = np.linalg.norm(self.spatial_coords - subhalo, axis=1)
                nearby = distances < radius_search
                if np.sum(nearby) > 0:
                    local_density = np.mean(self.predictions[nearby])
                else:
                    local_density = 0
                subhalo_local_density.append(local_density)
            
            subhalo_local_density = np.array(subhalo_local_density)
            
            fig.add_trace(
                go.Scatter(
                    x=subhalo_distances,
                    y=subhalo_local_density,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=subhalo_local_density,
                        colorscale='Plasma',
                        opacity=0.7
                    ),
                    name="Local Density"
                ),
                row=1, col=2
            )
            
            # Add best fit trend line
            mask = subhalo_local_density > 0
            if np.sum(mask) > 1:  # Need at least 2 points for regression
                coeffs = np.polyfit(subhalo_distances[mask], np.log10(subhalo_local_density[mask]), 1)
                x_trend = np.linspace(min(subhalo_distances), max(subhalo_distances), 100)
                y_trend = 10 ** (coeffs[0] * x_trend + coeffs[1])
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name="Density Trend"
                    ),
                    row=1, col=2
                )
            
            # 3. Subhalo Occupation
            # Compute subhalo count in radial shells
            r_bins = np.linspace(0, max(subhalo_distances), 20)
            subhalo_counts = np.zeros(len(r_bins)-1)
            
            for i in range(len(r_bins)-1):
                mask = (subhalo_distances >= r_bins[i]) & (subhalo_distances < r_bins[i+1])
                subhalo_counts[i] = np.sum(mask)
            
            r_centers = (r_bins[1:] + r_bins[:-1]) / 2
            
            fig.add_trace(
                go.Scatter(
                    x=r_centers,
                    y=subhalo_counts,
                    mode='lines+markers',
                    line=dict(color='purple', width=2),
                    marker=dict(size=8),
                    name="Subhalo Count"
                ),
                row=2, col=1
            )
            
            # 4. Clustering Metrics
            # If we have time information, show clustering evolution
            if self.cosmological_time is not None:
                times = np.sort(np.unique(self.cosmological_time))
                clustering_metric = []
                
                for t in times:
                    mask = self.cosmological_time == t
                    if np.sum(mask) > 0:
                        # Simple clustering metric: standard deviation of positions
                        std_positions = np.std(self.spatial_coords[mask], axis=0)
                        clustering_value = np.mean(std_positions)  # Average over 3 dimensions
                        clustering_metric.append(clustering_value)
                    else:
                        clustering_metric.append(np.nan)  # No data for this time
                
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=clustering_metric,
                        mode='lines+markers',
                        line=dict(color='teal', width=2),
                        marker=dict(size=8),
                        name="Clustering Metric"
                    ),
                    row=2, col=2
                )
            elif self.redshifts is not None:
                # Use redshifts if no cosmological time available
                redshifts = np.sort(np.unique(self.redshifts))
                clustering_metric = []
                
                for z in redshifts:
                    mask = self.redshifts == z
                    if np.sum(mask) > 0:
                        std_positions = np.std(self.spatial_coords[mask], axis=0)
                        clustering_value = np.mean(std_positions)
                        clustering_metric.append(clustering_value)
                    else:
                        clustering_metric.append(np.nan)
                
                fig.add_trace(
                    go.Scatter(
                        x=redshifts,
                        y=clustering_metric,
                        mode='lines+markers',
                        line=dict(color='teal', width=2),
                        marker=dict(size=8),
                        name="Clustering Metric"
                    ),
                    row=2, col=2
                )
            else:
                # Add a placeholder text if no time information available
                fig.add_annotation(
                    text="No time information available",
                    xref="x4", yref="y4",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=r"$\text{Hierarchical Structure Analysis of Dark Matter Halos}$",
                width=1200,
                height=1000,
                template='plotly_white',
                scene=dict(
                    xaxis_title=r"$X \, (\text{Mpc})$",
                    yaxis_title=r"$Y \, (\text{Mpc})$",
                    zaxis_title=r"$Z \, (\text{Mpc})$",
                    aspectmode='cube',
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                )
            )
            
            # Update axes titles for subplots
            fig.update_xaxes(title_text="Distance from Center (Mpc)", row=1, col=2)
            fig.update_yaxes(title_text="Local DM Density", row=1, col=2)
            
            fig.update_xaxes(title_text="Radius (Mpc)", row=2, col=1)
            fig.update_yaxes(title_text="Subhalo Count", row=2, col=1)
            
            if self.cosmological_time is not None:
                fig.update_xaxes(title_text="Cosmological Time (Gyr)", row=2, col=2)
            else:
                fig.update_xaxes(title_text="Redshift", row=2, col=2)
            fig.update_yaxes(title_text="Clustering Metric", row=2, col=2)
            
            # Save outputs
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(self.save_dir, 'hierarchical_structure.html')
                fig.write_html(path, include_mathjax='cdn')
                logger.info(f"Hierarchical structure visualization saved to {path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in hierarchical structure visualization: {e}", exc_info=True)
            return None
    
    def create_visualization_gif(self, fig, frames, save_path):
        """
        Create an animated GIF from a figure with frames.

        Args:
            fig (plotly.graph_objects.Figure): Figure with frames
            frames (list): List of frames
            save_path (str): Path to save the GIF
        """
        try:
            # Convert frames to images
            image_frames = []
            for frame in frames:
                # Create a copy of the figure
                fig_copy = go.Figure(fig)
                
                # Update with frame data
                for i, trace in enumerate(frame.data):
                    if i < len(fig_copy.data):
                        fig_copy.data[i].update(trace)
                
                # Update title if present
                if hasattr(frame, 'name') and frame.name:
                    current_title = fig_copy.layout.title.text
                    if current_title:
                        base_title = current_title.split(" at")[0]
                        fig_copy.update_layout(title=f"{base_title} at {frame.name}")
                
                # Convert to image
                img_bytes = fig_copy.to_image(format="png", width=1200, height=800)
                img = Image.open(io.BytesIO(img_bytes))
                image_frames.append(np.array(img))
            
            # Save as GIF
            imageio.mimsave(save_path, image_frames, duration=300)
            logger.info(f"Visualization GIF saved to {save_path}")
        except Exception as e:
            logger.error(f"Error creating GIF: {e}", exc_info=True)
                    
    def create_evolution_gif(self, save_path):
        """Create a GIF of the evolution of the model
        
        Args:
        save_path (str): Path to save the GIF
        """
        # First check if we have time or redshift data
        if self.cosmological_time is not None:
            time_variable = self.cosmological_time
            frame_prefix = "t="
            unique_times = np.sort(np.unique(time_variable))
        elif self.redshifts is not None:
            time_variable = self.redshifts
            frame_prefix = "z="
            unique_times = np.sort(np.unique(time_variable))[::-1]  # Reverse for redshift
        else:
            logger.error("No time or redshift data available for evolution GIF")
            return
        
        frames = []
        
        for t in unique_times:
            mask = (time_variable == t)
            
            if np.sum(mask) == 0:
                continue  # Skip times with no data
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=self.spatial_coords[mask, 0],
                    y=self.spatial_coords[mask, 1],
                    z=self.spatial_coords[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=self.predictions[mask],
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title=r"$\text{Density}$")
                    )
                )
            ])
        
            # Format time value for display
            if frame_prefix == "z=":
                time_display = f"{t:.2f}"
            else:
                time_display = f"{t:.2f}"
            
            fig.update_layout(
                title=r"$\text{Dark Matter Distribution at } " + frame_prefix + time_display + "$",
                width=800,
                height=800,
                template='plotly_white',
                scene=dict(
                    xaxis_title=r"$X \, (\text{Mpc})$",
                    yaxis_title=r"$Y \, (\text{Mpc})$",
                    zaxis_title=r"$Z \, (\text{Mpc})$",
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                )
            )
            
            # Convert to image
            img_bytes = fig.to_image(format="png")
            img = Image.open(io.BytesIO(img_bytes))
            frames.append(np.array(img))
        
        # Save as GIF
        imageio.mimsave(save_path, frames, duration=100)
        logger.info(f"Evolution GIF saved to {save_path}")

    def visualize_cosmic_web(self):
        """
        Visualize the cosmic web structure with filaments connecting dense regions.
        This is a basic implementation showing connections between high-density regions.
        
        Returns:
            plotly.graph_objects.Figure: The visualization figure or None if error occurs
        """
        try:
            # Identify high-density regions
            threshold = np.percentile(self.predictions, 90)  # Top 10% density regions
            high_density_mask = self.predictions > threshold
            high_density_coords = self.spatial_coords[high_density_mask]
            
            if len(high_density_coords) < 2:
                logger.warning("Not enough high-density regions identified.")
                return None
            
            # For performance, limit the number of points if too many
            if len(high_density_coords) > 200:
                # Sample a subset of high-density points
                rng = np.random.RandomState(42)
                indices = rng.choice(len(high_density_coords), 200, replace=False)
                high_density_coords = high_density_coords[indices]
            
            # Create a figure
            fig = go.Figure()
            
            # Add all data points with lower opacity
            fig.add_trace(
                go.Scatter3d(
                    x=self.spatial_coords[:, 0],
                    y=self.spatial_coords[:, 1],
                    z=self.spatial_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=self.predictions,
                        colorscale='Viridis',
                        opacity=0.3,
                        colorbar=dict(title="Density")
                    ),
                    name="All Particles"
                )
            )
            
            # Add high-density regions
            fig.add_trace(
                go.Scatter3d(
                    x=high_density_coords[:, 0],
                    y=high_density_coords[:, 1],
                    z=high_density_coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='red',
                        opacity=0.8
                    ),
                    name="High Density"
                )
            )
            
            # Add connections (filaments) between nearby high-density regions
            # This is a simplified approach; real cosmic web filament detection is more complex
            max_distance = np.mean(np.linalg.norm(self.spatial_coords, axis=1)) * 0.5  # Adjust coefficient as needed
            
            for i in range(len(high_density_coords)):
                for j in range(i+1, len(high_density_coords)):
                    point1 = high_density_coords[i]
                    point2 = high_density_coords[j]
                    distance = np.linalg.norm(point1 - point2)
                    
                    if distance < max_distance:
                        # Draw a line between these two points
                        fig.add_trace(
                            go.Scatter3d(
                                x=[point1[0], point2[0]],
                                y=[point1[1], point2[1]],
                                z=[point1[2], point2[2]],
                                mode='lines',
                                line=dict(
                                    color='rgba(255, 165, 0, 0.6)',  # Orange with transparency
                                    width=2
                                ),
                                showlegend=False
                            )
                        )
            
            # Update layout
            fig.update_layout(
                title="Cosmic Web Structure with Filaments",
                scene=dict(
                    xaxis_title="X (Mpc)",
                    yaxis_title="Y (Mpc)",
                    zaxis_title="Z (Mpc)",
                    aspectmode='cube',
                    xaxis=dict(range=self.x_limits),
                    yaxis=dict(range=self.y_limits),
                    zaxis=dict(range=self.z_limits)
                ),
                width=1000,
                height=800,
                template='plotly_white'
            )
            
            # Save outputs
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                path = os.path.join(self.save_dir, 'cosmic_web.html')
                fig.write_html(path, include_mathjax='cdn')
                logger.info(f"Cosmic web visualization saved to {path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in cosmic web visualization: {e}", exc_info=True)
            return None
</antA


import os
import csv
import datetime
import numpy as np
import torch as tc
import torch.nn as tcn
import torch.optim as tco
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model_metrics import CustomMetrics
from custom_dataset import DMDataset
from rnn_model import RNNDMHaloMapper
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

class ModelTrainer:
    def __init__(self, model=None, criterion=None, optimizer=None, patience=10, device=None, save_dir=None):
        self.model = model if model is not None else RNNDMHaloMapper()
        self.device = device if device else ('cuda' if tc.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.criterion = criterion or tc.nn.MSELoss()
        self.optimizer = optimizer or tc.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        self.save_dir = save_dir
        self.metrics_calculator = CustomMetrics()
        self.best_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        self.train_loader = None
        self.val_loader = None

        self.scaler = GradScaler()

    def set_loaders(self, X, y, batch_size=128):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
        train_dataset = DMDataset(X_train, y_train)
        val_dataset = DMDataset(X_val, y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    def __create_save_directory(self, model_name):
        model_dir = os.path.join(self.save_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(model_dir, exist_ok=True)
        return {
            'model_dir': model_dir,
            'best_model_path': os.path.join(model_dir, "best_model.pth"),
            'train_history_path': os.path.join(model_dir, "train_history.npy"),
            'train_history_file': os.path.join(model_dir, "train_history.csv")
        }

    def __train_epoch(self):
        self.model.train()
        total_loss = 0
        metrics = {k: 0 for k in ['mse', 'rmse', 'mae', 'r2', 'gaussian_nll', 'poisson_nll']}

        for i, (x, y) in enumerate(tqdm(self.train_loader, desc="Training Batches")):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            with tc.no_grad():
                ytrue = y.detach().cpu().numpy()
                ypred = output.detach().cpu().numpy()
                batch_metrics = self.metrics_calculator.calculate_all_metrics(ytrue, ypred)
                for k, v in batch_metrics.items():
                    metrics[k] += v

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in metrics.items()}

    def __validate_epoch(self):
        self.model.eval()
        total_loss = 0
        metrics = {k: 0 for k in ['mse', 'rmse', 'mae', 'r2', 'gaussian_nll', 'poisson_nll']}

        with tc.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation Batches"):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)
                total_loss += loss.item()

                batch_metrics = self.metrics_calculator.calculate_all_metrics(y.cpu(), output.cpu())
                for k, v in batch_metrics.items():
                    metrics[k] += v

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in metrics.items()}

    def __save_training_progress(self, epoch, train_loss, val_loss, file_path):
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            writer.writerow([epoch + 1, train_loss, val_loss])

    def train(self, num_epochs, model_name, train_loader=None, val_loader=None, X=None, y=None, batch_size=128):
        if train_loader and val_loader:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            self.set_loaders(X, y, batch_size=batch_size)

        save_paths = self.__create_save_directory(model_name)
        scheduler = tco.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            train_loss, train_metrics = self.__train_epoch()
            val_loss, val_metrics = self.__validate_epoch()

            scheduler.step(val_loss)

            self.__save_training_progress(epoch, train_loss, val_loss, save_paths['train_history_file'])

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)

            print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
            if (epoch + 1) % 5 == 0:
                print("Train Metrics:", train_metrics)
                print("Val Metrics:", val_metrics)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                tc.save(self.model.state_dict(), save_paths['best_model_path'])
                print(f"New best model saved at epoch {epoch + 1}.")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        np.save(save_paths['train_history_path'], self.history)
        print("Training completed.")
        return self.history, save_paths['model_dir']

import os
import csv
import datetime
import numpy as np
import torch as tc
import torch.nn as tcn
import torch.optim as tco
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model_metrics import CustomMetrics
from custom_dataset import DMDataset
from rnn_model import RNNDMHaloMapper
from directory_manager import PathManager

path_manager = PathManager()
logger = path_manager.get_logger()

class ModelTrainer:
    def __init__(self, model=None, criterion=None, optimizer=None, patience=10, device=None, save_dir=None):
        self.model = model if model is not None else RNNDMHaloMapper()
        self.device = device if device else ('cuda' if tc.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.criterion = criterion or tc.nn.MSELoss()
        self.optimizer = optimizer or tc.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        self.save_dir = save_dir
        self.metrics_calculator = CustomMetrics()
        self.best_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        self.train_loader = None
        self.val_loader = None

        self.scaler = GradScaler()

    def set_loaders(self, X, y, batch_size=128):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
        train_dataset = DMDataset(X_train, y_train)
        val_dataset = DMDataset(X_val, y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    def __create_save_directory(self, model_name):
        model_dir = os.path.join(self.save_dir, f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(model_dir, exist_ok=True)
        return {
            'model_dir': model_dir,
            'best_model_path': os.path.join(model_dir, "best_model.pth"),
            'train_history_path': os.path.join(model_dir, "train_history.npy"),
            'train_history_file': os.path.join(model_dir, "train_history.csv")
        }

    def __train_epoch(self):
        self.model.train()
        total_loss = 0
        metrics = {k: 0 for k in ['mse', 'rmse', 'mae', 'r2', 'gaussian_nll', 'poisson_nll']}

        for i, (x, y) in enumerate(tqdm(self.train_loader, desc="Training Batches")):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            with tc.no_grad():
                ytrue = y.detach().cpu().numpy()
                ypred = output.detach().cpu().numpy()
                batch_metrics = self.metrics_calculator.calculate_all_metrics(ytrue, ypred)
                for k, v in batch_metrics.items():
                    metrics[k] += v

        n = len(self.train_loader)
        return total_loss / n, {k: v / n for k, v in metrics.items()}

    def __validate_epoch(self):
        self.model.eval()
        total_loss = 0
        metrics = {k: 0 for k in ['mse', 'rmse', 'mae', 'r2', 'gaussian_nll', 'poisson_nll']}

        with tc.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation Batches"):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.criterion(output, y)
                total_loss += loss.item()

                batch_metrics = self.metrics_calculator.calculate_all_metrics(y.cpu(), output.cpu())
                for k, v in batch_metrics.items():
                    metrics[k] += v

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in metrics.items()}

    def __save_training_progress(self, epoch, train_loss, val_loss, file_path):
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            writer.writerow([epoch + 1, train_loss, val_loss])

    def train(self, num_epochs, model_name, train_loader=None, val_loader=None, X=None, y=None, batch_size=128):
        if train_loader and val_loader:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            self.set_loaders(X, y, batch_size=batch_size)

        save_paths = self.__create_save_directory(model_name)
        scheduler = tco.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            train_loss, train_metrics = self.__train_epoch()
            val_loss, val_metrics = self.__validate_epoch()

            scheduler.step(val_loss)

            self.__save_training_progress(epoch, train_loss, val_loss, save_paths['train_history_file'])

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)

            print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
            if (epoch + 1) % 5 == 0:
                print("Train Metrics:", train_metrics)
                print("Val Metrics:", val_metrics)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                tc.save(self.model.state_dict(), save_paths['best_model_path'])
                print(f"New best model saved at epoch {epoch + 1}.")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        np.save(save_paths['train_history_path'], self.history)
        print("Training completed.")
        return self.history, save_paths['model_dir']
