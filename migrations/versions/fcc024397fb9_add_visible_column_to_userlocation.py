"""Add visible column to UserLocation

Revision ID: fcc024397fb9
Revises: a3b0907dd11a
Create Date: 2025-03-24 19:25:20.841465

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fcc024397fb9'
down_revision = 'a3b0907dd11a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user_location', schema=None) as batch_op:
        batch_op.add_column(sa.Column('visible', sa.Boolean(), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user_location', schema=None) as batch_op:
        batch_op.drop_column('visible')

    # ### end Alembic commands ###
