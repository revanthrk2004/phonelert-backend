"""Remove password field

Revision ID: 0279a4d8ddd1
Revises: 6ab55cf033eb
Create Date: 2025-03-23 05:11:13.415750

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0279a4d8ddd1'
down_revision = '6ab55cf033eb'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column('password_hash',
               existing_type=sa.VARCHAR(length=256),
               type_=sa.String(length=512),
               existing_nullable=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column('password_hash',
               existing_type=sa.String(length=512),
               type_=sa.VARCHAR(length=256),
               existing_nullable=False)

    # ### end Alembic commands ###
