"""Added username to User model

Revision ID: d0f97c792638
Revises: 8f3cc9169220
Create Date: 2025-03-15 17:38:36.539608

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd0f97c792638'
down_revision = '8f3cc9169220'
branch_labels = None
depends_on = None


def upgrade():
    # ✅ 1. Add 'username' column ALLOWING NULL TEMPORARILY
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column("username", sa.String(length=100), nullable=True))  

    # ✅ 2. Update existing rows with a default username (PREVENT NULL)
    op.execute("UPDATE \"user\" SET username = 'UnknownUser' WHERE username IS NULL")

    # ✅ 3. Now make the 'username' column NOT NULL
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column("username", existing_type=sa.String(length=100), nullable=False)

    # ✅ 4. Drop 'secondary_device_token' since it's no longer needed
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column("secondary_device_token")


def downgrade():
    # ✅ Rollback changes: Restore 'secondary_device_token' & remove 'username'
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('secondary_device_token', sa.VARCHAR(length=255), nullable=True))
        batch_op.drop_column('username')
