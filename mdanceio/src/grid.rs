use cgmath::{Vector2, Vector3};
use wgpu::util::DeviceExt;

use crate::{forward::LineVertexUnit, line_drawer::LineDrawer, project::Project};

pub struct Grid {
    // TODO
    line_drawer: LineDrawer,
    vertex_buffer: wgpu::Buffer,
    line_color: Vector3<f32>,
    cell: Vector2<f32>,
    size: Vector2<f32>,
    opacity: f32,
    visible: bool,
}

impl Grid {
    pub fn new(texture_format: wgpu::TextureFormat, device: &wgpu::Device) -> Self {
        let line_color = Vector3::new(0.5f32, 0.5f32, 0.5f32);
        let cell = Vector2::new(5f32, 5f32);
        let size = Vector2::new(10f32, 10f32);
        let opacity = 1.0f32;
        let vertices = Self::build_vertices(line_color, cell, size, opacity);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GridPass/Vertices"),
            contents: bytemuck::cast_slice(&vertices[..]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            line_drawer: LineDrawer::new(&vertices, texture_format, device),
            vertex_buffer,
            line_color,
            cell,
            size,
            opacity,
            visible: true,
        }
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, value: bool) {
        if value != self.visible {
            self.visible = value;
            // TODO: publish event
        }
    }

    pub fn num_vertices(&self) -> u16 {
        (self.size.x + self.size.y + 1.0) as u16 * 4u16 + 6u16
    }

    pub fn draw(
        &self,
        color_view: &wgpu::TextureView,
        _depth_view: Option<&wgpu::TextureView>,
        project: &Project,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if self.visible {
            let (view, projection) = project.active_camera().get_view_transform();
            self.line_drawer.update_uniform(
                projection * view,
                self.line_color.extend(1.0f32),
                queue,
            );
            let vertices = self.build_grid_vertices();
            self.line_drawer.update_vertex_buffer(&vertices, queue);
            self.line_drawer.draw(color_view, device, queue);
        }
    }

    fn build_grid_vertices(&self) -> Vec<LineVertexUnit> {
        Self::build_vertices(self.line_color, self.cell, self.size, self.opacity)
    }

    fn build_vertices(
        line_color: Vector3<f32>,
        cell: Vector2<f32>,
        size: Vector2<f32>,
        opacity: f32,
    ) -> Vec<LineVertexUnit> {
        let mut vertices = vec![];
        let color = line_color
            .map(|v| (v * (0xffu32 as f32)) as u8)
            .extend((opacity * (0xffu32 as f32)) as u8)
            .into();
        for i in -(size.x as i32)..=(size.x as i32) {
            let height = size.y * cell.y;
            let x = (i as f32) * cell.x;
            vertices.push(LineVertexUnit {
                position: [x, 0f32, height],
                color,
            });
            vertices.push(LineVertexUnit {
                position: [x, 0f32, if i == 0 { 0f32 } else { -height }],
                color,
            });
        }
        for i in -(size.y as i32)..=(size.y as i32) {
            let width = size.y * cell.y;
            let z = (i as f32) * cell.y;
            vertices.push(LineVertexUnit {
                position: [-width, 0f32, z],
                color,
            });
            vertices.push(LineVertexUnit {
                position: [if i == 0 { 0f32 } else { width }, 0f32, z],
                color,
            });
        }
        for c in &[
            Vector3::<f32>::unit_x(),
            Vector3::unit_y(),
            Vector3::unit_z(),
        ] {
            let color = c
                .map(|v| (v * (0xffu32 as f32)) as u8)
                .extend((opacity * (0xffu32 as f32)) as u8)
                .into();
            let width = size.x * cell.x;
            vertices.push(LineVertexUnit {
                position: [0f32, 0f32, 0f32],
                color,
            });
            vertices.push(LineVertexUnit {
                position: c
                    .map(|v| (v as f32) * width * (if c.z > 0f32 { -1f32 } else { 1f32 }))
                    .into(),
                color,
            });
        }
        vertices
    }
}
