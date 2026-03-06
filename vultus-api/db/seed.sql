INSERT INTO users (email, password_hash, full_name, role)
VALUES (
  'admin@vultus.ai',
  '$2a$10$nMwzXU2KRwN0NXFyv0uqj.ecVp2nnpGu76cjO2cnvzeGU0D9YvNmu',
  'Vultus Admin',
  'admin'
) ON CONFLICT DO NOTHING;

-- Password for the seeded admin is: Admin@123
